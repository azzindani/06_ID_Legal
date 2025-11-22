"""
Enhanced KG Dataset Loader with Centralized Logging
All logs go to the same centralized log file
"""

import sqlite3
import numpy as np
import json
import zlib
import gc
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Import centralized logging system
from logger_utils import get_logger, ProgressTracker


class EnhancedKGDatasetLoader:
    """
    SQLite-based dataset loader with centralized logging
    All logs are written to the same system log file
    """
    
    def __init__(self, dataset_name: str, embedding_dim: int):
        self.dataset_name = dataset_name
        self.embedding_dim = embedding_dim
        self.db_path = None
        self.conn = None
        
        # Get logger from centralized system
        self.logger = get_logger("DataLoader")
        
        # Data storage
        self.all_records = []
        self.embeddings = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        
        # KG indexes
        self.kg_entities_lookup = {}
        self.kg_cross_references_lookup = {}
        self.kg_domains_lookup = {}
        self.kg_concept_clusters_lookup = {}
        self.kg_legal_actions_lookup = {}
        self.kg_sanctions_lookup = {}
        self.kg_concept_vectors_lookup = {}
        
        # Numeric indexes
        self.authority_index = {}
        self.temporal_index = {}
        self.kg_connectivity_index = {}
        self.hierarchy_index = {}
        self.domain_index = {}
        
        self.logger.info("DataLoader initialized", {
            "dataset": dataset_name,
            "embedding_dim": embedding_dim
        })

    def load(self, progress_callback: Optional[Callable[[str], None]] = None) -> dict:
        """
        Load dataset and return data in expected format.

        Args:
            progress_callback: Optional callback function for progress updates

        Returns:
            dict: Dictionary with 'documents' key containing all records
        """
        success = self.load_from_huggingface(progress_callback)
        if success:
            return {'documents': self.all_records}
        return {'documents': []}

    def load_from_huggingface(self, progress_callback: Optional[Callable[[str], None]] = None):
        """
        Load SQLite database from HuggingFace repository
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Starting HuggingFace dataset download")
            
            if progress_callback:
                progress_callback("🔥 Downloading SQLite database from HuggingFace...")
            
            from huggingface_hub import hf_hub_download
            
            # Download database file
            try:
                self.logger.debug("Initiating HF download", {
                    "repo": self.dataset_name,
                    "file": "id_regulations.db"
                })
                
                self.db_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="id_regulations.db",
                    repo_type="dataset"
                )
                
                self.logger.success("Database downloaded successfully", {
                    "path": self.db_path
                })
                
            except Exception as e:
                self.logger.error("Failed to download database", {
                    "repo": self.dataset_name,
                    "file": "id_regulations.db",
                    "error": str(e)
                })
                
                if progress_callback:
                    progress_callback(f"❌ Failed to download database: {str(e)}")
                
                raise Exception(f"HuggingFace download failed: {str(e)}")
            
            if progress_callback:
                progress_callback(f"✅ Database downloaded: {self.db_path}")
            
            # Connect to database
            try:
                self.logger.debug("Connecting to SQLite database")
                
                self.conn = sqlite3.connect(self.db_path)
                self.conn.execute("PRAGMA query_only = ON")
                self.conn.execute("PRAGMA cache_size=-64000")
                self.conn.execute("PRAGMA mmap_size=268435456")
                
                self.logger.info("Database connection established")
                
            except Exception as e:
                self.logger.error("Failed to connect to database", {
                    "error": str(e)
                })
                
                if progress_callback:
                    progress_callback(f"❌ Failed to connect to database: {str(e)}")
                
                raise Exception(f"Database connection failed: {str(e)}")
            
            # Verify database structure
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.logger.info("Database tables discovered", {
                "tables": ", ".join(tables),
                "count": len(tables)
            })
            
            if progress_callback:
                progress_callback(f"📊 Found tables: {', '.join(tables)}")
            
            if 'regulations' not in tables:
                self.logger.error("Required table 'regulations' not found")
                raise Exception("Table 'regulations' not found in database!")
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM regulations")
            total_rows = cursor.fetchone()[0]
            
            if total_rows == 0:
                self.logger.error("regulations table is empty")
                raise Exception("regulations table is empty!")
            
            self.logger.info("Starting record processing", {
                "total_records": f"{total_rows:,}"
            })
            
            if progress_callback:
                progress_callback(f"📊 Processing {total_rows:,} records...")
            
            # Load records in chunks
            chunk_size = 5000
            all_records_temp = []
            embeddings_temp = []
            
            for offset in range(0, total_rows, chunk_size):
                chunk_num = (offset // chunk_size) + 1
                total_chunks = (total_rows + chunk_size - 1) // chunk_size
                
                self.logger.debug(f"Processing chunk {chunk_num}/{total_chunks}", {
                    "offset": offset,
                    "chunk_size": chunk_size
                })
                
                # Fetch main records
                cursor.execute(f"""
                    SELECT 
                        global_id, local_id, regulation_type, enacting_body,
                        regulation_number, year, about, effective_date,
                        chapter, article, content, chunk_id,
                        kg_entity_count, kg_cross_ref_count, kg_primary_domain,
                        kg_domain_confidence, kg_cluster_count, kg_cluster_diversity,
                        kg_authority_score, kg_hierarchy_level, kg_temporal_score,
                        kg_years_old, kg_legal_richness, kg_legal_complexity,
                        kg_has_obligations, kg_has_prohibitions, kg_has_permissions,
                        kg_completeness_score, kg_connectivity_score,
                        kg_pagerank, kg_degree_centrality
                    FROM regulations
                    LIMIT {chunk_size} OFFSET {offset}
                """)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    record = self._row_to_record(row)
                    all_records_temp.append(record)
                
                # Fetch embeddings for this chunk
                global_ids = [row[0] for row in rows]
                placeholders = ','.join(['?' for _ in global_ids])
                
                # Check if embeddings table exists
                if 'embeddings' in tables:
                    cursor.execute(f"""
                        SELECT global_id, embedding, dimension
                        FROM embeddings
                        WHERE global_id IN ({placeholders})
                    """, global_ids)
                    
                    embedding_rows = cursor.fetchall()
                    embedding_dict = {row[0]: (row[1], row[2]) for row in embedding_rows}
                else:
                    if offset == 0:
                        self.logger.warning("Embeddings table not found, using zero vectors")
                        if progress_callback:
                            progress_callback("   ⚠️ Embeddings table not found, using zero vectors")
                    embedding_dict = {}
                
                # Add embeddings in correct order
                for global_id in global_ids:
                    if global_id in embedding_dict:
                        emb_blob, dim = embedding_dict[global_id]
                        embedding = self._decompress_embedding(emb_blob, dim)
                        embeddings_temp.append(embedding)
                    else:
                        embeddings_temp.append(np.zeros(self.embedding_dim, dtype=np.float32))
                
                # Memory cleanup
                del rows, embedding_rows, embedding_dict
                gc.collect()
            
            self.all_records = all_records_temp
            
            self.logger.success("Records loaded successfully", {
                "total_records": f"{len(self.all_records):,}"
            })
            
            if progress_callback:
                progress_callback("📊 Converting embeddings to tensor...")
            
            # Convert embeddings to tensor
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.logger.info("Converting embeddings to tensor", {
                "device": str(device),
                "shape": f"({len(embeddings_temp)}, {self.embedding_dim})"
            })
            
            self.embeddings = torch.tensor(
                np.array(embeddings_temp, dtype=np.float32),
                device=device
            )
            
            self.logger.success("Embeddings converted", {
                "shape": str(self.embeddings.shape),
                "device": str(device)
            })
            
            del embeddings_temp
            gc.collect()
            
            if progress_callback:
                progress_callback("📄 Loading TF-IDF vectors...")
            
            # Load TF-IDF (sparse format)
            self._load_tfidf_sparse()
            
            if progress_callback:
                progress_callback("🗃️ Building KG indexes...")
            
            # Build KG indexes
            self._build_enhanced_kg_indexes()
            
            # Close connection (will reopen for queries)
            if self.conn:
                self.conn.close()
                self.conn = None
            
            gc.collect()
            
            kg_count = len(self.kg_entities_lookup)
            
            self.logger.success("Dataset loading completed", {
                "total_records": f"{len(self.all_records):,}",
                "kg_enhanced": f"{kg_count:,}",
                "enhancement_rate": f"{kg_count/len(self.all_records)*100:.1f}%"
            })
            
            if progress_callback:
                progress_callback(f"✅ Ready: {len(self.all_records):,} records with {kg_count:,} KG-enhanced")
            
            return True
            
        except Exception as e:
            self.logger.error("Loading failed", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            if progress_callback:
                progress_callback(f"❌ Loading failed: {str(e)}")
            
            import traceback
            self.logger.debug("Full traceback", {
                "traceback": traceback.format_exc()[:500]
            })
            traceback.print_exc()
            
            # Clean up on error
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
                self.conn = None
            
            return False
    
    def _row_to_record(self, row) -> Dict:
        """Convert database row to record dictionary"""
        return {
            'global_id': row[0],
            'local_id': row[1],
            'regulation_type': str(row[2]) if row[2] else 'Unknown',
            'enacting_body': str(row[3]) if row[3] else 'Unknown',
            'regulation_number': str(row[4]) if row[4] else 'N/A',
            'year': str(row[5]) if row[5] else '2023',
            'about': str(row[6])[:200] if row[6] else '',
            'effective_date': str(row[7]) if row[7] else '2023-01-01',
            'chapter': str(row[8]) if row[8] else 'N/A',
            'article': str(row[9]) if row[9] else 'N/A',
            'content': str(row[10])[:500] if row[10] else '',
            'chunk_id': int(row[11]) if row[11] else 1,
            
            # KG features
            'kg_entity_count': int(row[12]) if row[12] else 0,
            'kg_cross_ref_count': int(row[13]) if row[13] else 0,
            'kg_primary_domain': str(row[14]) if row[14] else 'Unknown',
            'kg_domain_confidence': float(row[15]) if row[15] else 0.0,
            'kg_cluster_count': int(row[16]) if row[16] else 0,
            'kg_cluster_diversity': float(row[17]) if row[17] else 0.0,
            'kg_authority_score': float(row[18]) if row[18] else 0.5,
            'kg_hierarchy_level': int(row[19]) if row[19] else 5,
            'kg_temporal_score': float(row[20]) if row[20] else 0.6,
            'kg_years_old': int(row[21]) if row[21] else 1,
            'kg_legal_richness': float(row[22]) if row[22] else 0.0,
            'kg_legal_complexity': float(row[23]) if row[23] else 0.0,
            'kg_has_obligations': bool(row[24]) if row[24] else False,
            'kg_has_prohibitions': bool(row[25]) if row[25] else False,
            'kg_has_permissions': bool(row[26]) if row[26] else False,
            'kg_completeness_score': float(row[27]) if row[27] else 0.0,
            'kg_connectivity_score': float(row[28]) if row[28] else 0.0,
            'kg_pagerank': float(row[29]) if row[29] else 0.0,
            'kg_degree_centrality': float(row[30]) if row[30] else 0.0,
            
            # Store as placeholders
            'kg_entities_json': '[]',
            'kg_cross_references_json': '[]',
            'kg_legal_domains_json': '[]',
            'kg_concept_clusters_json': '{}',
            'kg_legal_actions_json': '{}',
            'kg_sanctions_json': '{}',
            'kg_concept_vector_json': '[]',
            'kg_citation_impact_json': '{}'
        }
    
    def _decompress_embedding(self, blob, dimension):
        """Decompress embedding from BLOB"""
        try:
            try:
                data = zlib.decompress(blob)
            except:
                data = blob
            
            embedding = np.frombuffer(data, dtype=np.float16)
            
            if len(embedding) != dimension:
                self.logger.warning("Embedding dimension mismatch", {
                    "expected": dimension,
                    "got": len(embedding)
                })
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                embedding = embedding.astype(np.float32)
            
            return embedding
            
        except Exception as e:
            self.logger.error("Error decompressing embedding", {
                "error": str(e)
            })
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _load_tfidf_sparse(self):
        """Load TF-IDF vectors in sparse format"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tfidf_vectors")
            count = cursor.fetchone()[0]
            
            if count == 0:
                self.logger.warning("TF-IDF table is empty, creating dummy vectorizer")
                self._create_dummy_vectorizer()
                return
            
            self.logger.info("Loading TF-IDF vectors", {
                "count": f"{count:,}"
            })
            
            # Load all TF-IDF vectors
            cursor.execute("SELECT global_id, tfidf_vector, dimension FROM tfidf_vectors ORDER BY rowid")
            
            all_sparse_rows = []
            tfidf_dim = None
            
            for row in cursor.fetchall():
                global_id, tfidf_blob, dimension = row
                
                if tfidf_dim is None:
                    tfidf_dim = dimension
                
                try:
                    data = zlib.decompress(tfidf_blob)
                except:
                    data = tfidf_blob
                
                sparse_data = json.loads(data.decode('utf-8'))
                
                sparse_row = sparse.csr_matrix(
                    (sparse_data['data'], sparse_data['indices'], sparse_data['indptr']),
                    shape=sparse_data['shape']
                )
                
                all_sparse_rows.append(sparse_row)
            
            self.tfidf_matrix = sparse.vstack(all_sparse_rows)
            self._create_working_vectorizer(tfidf_dim)
            
            self.logger.success("TF-IDF vectors loaded", {
                "shape": str(self.tfidf_matrix.shape),
                "dimension": tfidf_dim
            })
            
        except Exception as e:
            self.logger.error("Error loading TF-IDF", {
                "error": str(e)
            })
            self._create_dummy_vectorizer()
    
    def _create_working_vectorizer(self, n_features):
        """Create working vectorizer for TF-IDF"""
        class WorkingVectorizer:
            def __init__(self, features):
                self.vocabulary_ = {}
                self._tfidf = True
                self.idf_ = None
                self.stop_words_ = set()
                self.n_features = features
            
            def transform(self, texts):
                from scipy.sparse import csr_matrix
                if isinstance(texts, str):
                    texts = [texts]
                return csr_matrix((len(texts), self.n_features))
        
        self.tfidf_vectorizer = WorkingVectorizer(n_features)
        self.logger.debug("Working vectorizer created", {
            "features": n_features
        })
    
    def _create_dummy_vectorizer(self):
        """Create dummy vectorizer when TF-IDF not available"""
        class DummyVectorizer:
            def __init__(self):
                self.vocabulary_ = {}
                self._tfidf = True
                self.idf_ = None
                self.stop_words_ = set()
                self.n_features = 20000
            
            def transform(self, texts):
                from scipy.sparse import csr_matrix
                if isinstance(texts, str):
                    texts = [texts]
                return csr_matrix((len(texts), self.n_features))
        
        self.tfidf_vectorizer = DummyVectorizer()
        self.logger.info("Dummy vectorizer created")
    
    def _build_enhanced_kg_indexes(self):
        """Build KG indexes with lazy loading from database"""
        self.logger.info("Building KG indexes")
        
        # Reconnect for KG data queries
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA query_only = ON")
        
        cursor = self.conn.cursor()
        
        # Load KG JSON data
        cursor.execute("""
            SELECT 
                global_id, 
                kg_entities_json,
                kg_cross_references_json,
                kg_legal_domains_json,
                kg_concept_clusters_json,
                kg_legal_actions_json,
                kg_sanctions_json,
                kg_concept_vector_json
            FROM kg_json_data
        """)
        
        processed_count = 0
        for row in cursor.fetchall():
            global_id = row[0]
            
            for idx, field_name in enumerate([
                'kg_entities_lookup',
                'kg_cross_references_lookup', 
                'kg_domains_lookup',
                'kg_concept_clusters_lookup',
                'kg_legal_actions_lookup',
                'kg_sanctions_lookup',
                'kg_concept_vectors_lookup'
            ], start=1):
                
                if row[idx]:
                    try:
                        try:
                            data = zlib.decompress(row[idx])
                        except:
                            data = row[idx]
                        
                        json_str = data.decode('utf-8')
                        lookup = getattr(self, field_name)
                        
                        if json_str not in ['[]', '{}', 'null']:
                            lookup[global_id] = json_str
                    except Exception:
                        continue
            
            processed_count += 1
        
        self.logger.info("KG JSON data loaded", {
            "processed": f"{processed_count:,}",
            "entities": len(self.kg_entities_lookup),
            "cross_refs": len(self.kg_cross_references_lookup)
        })
        
        # Build numeric indexes
        self.authority_index = defaultdict(list)
        self.temporal_index = defaultdict(list)
        self.kg_connectivity_index = defaultdict(list)
        self.hierarchy_index = defaultdict(list)
        self.domain_index = defaultdict(list)
        
        for i, record in enumerate(self.all_records):
            try:
                authority_tier = max(0, min(10, int(record['kg_authority_score'] * 10)))
                self.authority_index[authority_tier].append(i)
                
                temporal_tier = max(0, min(10, int(record['kg_temporal_score'] * 10)))
                self.temporal_index[temporal_tier].append(i)
                
                kg_tier = max(0, min(10, int(record['kg_connectivity_score'] * 10)))
                self.kg_connectivity_index[kg_tier].append(i)
                
                hierarchy_tier = max(1, min(10, record['kg_hierarchy_level']))
                self.hierarchy_index[hierarchy_tier].append(i)
                
                domain = record['kg_primary_domain']
                self.domain_index[domain].append(i)
                
            except Exception:
                continue
        
        # Convert to dict
        self.authority_index = dict(self.authority_index)
        self.temporal_index = dict(self.temporal_index)
        self.kg_connectivity_index = dict(self.kg_connectivity_index)
        self.hierarchy_index = dict(self.hierarchy_index)
        self.domain_index = dict(self.domain_index)
        
        self.logger.success("KG indexes built", {
            "authority_tiers": len(self.authority_index),
            "temporal_tiers": len(self.temporal_index),
            "unique_domains": len(self.domain_index)
        })
        
        # Close connection
        self.conn.close()
        self.conn = None
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.all_records:
            self.logger.warning("No records available for statistics")
            return {}
        
        try:
            total_records = len(self.all_records)
            kg_enhanced = len(self.kg_entities_lookup)
            
            authority_scores = [r['kg_authority_score'] for r in self.all_records]
            temporal_scores = [r['kg_temporal_score'] for r in self.all_records]
            connectivity_scores = [r['kg_connectivity_score'] for r in self.all_records]
            entity_counts = [r['kg_entity_count'] for r in self.all_records]
            cross_ref_counts = [r['kg_cross_ref_count'] for r in self.all_records]
            
            stats = {
                'total_records': total_records,
                'kg_enhanced': kg_enhanced,
                'kg_enhancement_rate': kg_enhanced / total_records if total_records > 0 else 0,
                'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
                'tfidf_shape': self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
                'tfidf_enabled': self.tfidf_matrix is not None,
                'memory_optimized': True,
                'authority_tiers': len(self.authority_index),
                'temporal_tiers': len(self.temporal_index),
                'kg_connectivity_tiers': len(self.kg_connectivity_index),
                'hierarchy_tiers': len(self.hierarchy_index),
                'unique_domains': len(self.domain_index),
                'avg_authority_score': np.mean(authority_scores) if authority_scores else 0,
                'avg_temporal_score': np.mean(temporal_scores) if temporal_scores else 0,
                'avg_connectivity_score': np.mean(connectivity_scores) if connectivity_scores else 0,
                'avg_entities_per_doc': np.mean(entity_counts) if entity_counts else 0,
                'avg_cross_refs_per_doc': np.mean(cross_ref_counts) if cross_ref_counts else 0,
                'has_obligations': sum(1 for r in self.all_records if r['kg_has_obligations']),
                'has_prohibitions': sum(1 for r in self.all_records if r['kg_has_prohibitions']),
                'has_permissions': sum(1 for r in self.all_records if r['kg_has_permissions'])
            }
            
            self.logger.info("Statistics generated successfully")
            return stats
            
        except Exception as e:
            self.logger.error("Error generating statistics", {
                "error": str(e)
            })
            return {'error': str(e)}