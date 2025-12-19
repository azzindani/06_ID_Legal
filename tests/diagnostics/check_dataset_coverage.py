"""
Simple Dataset Coverage Check

Analyzes if the dataset contains relevant documents for tax objection queries
without requiring full pipeline initialization.

Usage:
    python tests/diagnostics/check_dataset_coverage.py
"""

import sys
import os
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_dataset():
    """Load dataset from JSON file"""
    print("Loading dataset...")

    # Try multiple possible locations
    possible_paths = [
        "data/legal_documents.json",
        "data/processed/legal_documents.json",
        "data/legal_data.json"
    ]

    for path in possible_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"   Found dataset at: {full_path}")
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"   ✓ Loaded {len(data)} documents")
            return data

    print("   ✗ Dataset not found. Searched:")
    for path in possible_paths:
        print(f"      - {path}")
    return None


def analyze_tax_law_coverage(dataset):
    """Analyze coverage of tax law documents"""
    print("\n" + "="*80)
    print("TAX LAW DOCUMENT COVERAGE ANALYSIS")
    print("="*80)

    # Key terms for tax objection query
    tax_terms = {
        "keberatan pajak": [],
        "UU KUP": [],
        "ketentuan umum perpajakan": [],
        "pengadilan pajak": [],
        "banding pajak": [],
        "wajib pajak": [],
        "sanksi pajak": [],
        "pajak penghasilan": [],
        "Pajak Pertambahan Nilai": [],
        "PPN": []
    }

    regulation_stats = defaultdict(int)
    year_stats = defaultdict(int)
    highly_relevant = []

    print("\nScanning dataset for tax law documents...")

    for doc in dataset:
        # Get document fields
        title = doc.get('regulation_title', '').lower()
        content = doc.get('content', '').lower()
        reg_type = doc.get('regulation_type', 'Unknown')
        reg_number = doc.get('regulation_number', 'N/A')
        year = doc.get('year', 'Unknown')

        regulation_stats[reg_type] += 1
        year_stats[year] += 1

        # Check for tax terms
        combined_text = title + " " + content
        matched_terms = []
        relevance_score = 0

        for term in tax_terms:
            if term in combined_text:
                tax_terms[term].append(doc)
                matched_terms.append(term)
                relevance_score += 1

        if relevance_score >= 2:  # At least 2 tax terms matched
            highly_relevant.append({
                'regulation_type': reg_type,
                'regulation_number': reg_number,
                'year': year,
                'title': doc.get('regulation_title', 'N/A'),
                'matched_terms': matched_terms,
                'relevance_score': relevance_score,
                'global_id': doc.get('global_id', 'N/A')
            })

    # Sort by relevance
    highly_relevant.sort(key=lambda x: x['relevance_score'], reverse=True)

    # Print results
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total documents: {len(dataset):,}")
    print(f"   Total regulation types: {len(regulation_stats)}")

    print(f"\n🔍 TAX TERM COVERAGE:")
    for term, docs in sorted(tax_terms.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(docs)
        percentage = (count / len(dataset)) * 100 if dataset else 0
        print(f"   '{term}': {count:,} docs ({percentage:.2f}%)")

    print(f"\n📋 TOP 15 REGULATION TYPES:")
    for reg_type, count in sorted(regulation_stats.items(), key=lambda x: x[1], reverse=True)[:15]:
        percentage = (count / len(dataset)) * 100
        print(f"   {reg_type}: {count:,} docs ({percentage:.1f}%)")

    print(f"\n📅 DOCUMENT YEARS:")
    sorted_years = sorted(year_stats.items(), key=lambda x: str(x[0]))
    for year, count in sorted_years[:20]:  # Show first 20 years
        print(f"   {year}: {count:,} docs")

    if highly_relevant:
        print(f"\n✅ HIGHLY RELEVANT TAX LAW DOCUMENTS: {len(highly_relevant)}")
        print(f"\nTop 20 most relevant documents for tax objection query:")
        print("-" * 80)

        for i, doc in enumerate(highly_relevant[:20], 1):
            print(f"\n{i}. [{doc['relevance_score']} terms] {doc['regulation_type']} No. {doc['regulation_number']}/{doc['year']}")
            print(f"   Title: {doc['title']}")
            print(f"   Matched: {', '.join(doc['matched_terms'])}")
            print(f"   Global ID: {doc['global_id']}")

    else:
        print(f"\n⚠️  WARNING: NO HIGHLY RELEVANT TAX DOCUMENTS FOUND!")
        print(f"   This explains why the RAG returns irrelevant results.")
        print(f"\n   The dataset may not contain:")
        print(f"   - UU KUP (Ketentuan Umum dan Tata Cara Perpajakan)")
        print(f"   - UU Pengadilan Pajak")
        print(f"   - Regulations about tax objections (keberatan pajak)")

    return highly_relevant


def check_specific_regulations(dataset):
    """Check for specific tax regulations"""
    print("\n" + "="*80)
    print("SPECIFIC TAX REGULATION CHECK")
    print("="*80)

    # Key tax regulations that should exist
    expected_regulations = [
        {"name": "UU KUP", "keywords": ["ketentuan umum", "tata cara perpajakan", "KUP"]},
        {"name": "UU Pengadilan Pajak", "keywords": ["pengadilan pajak", "banding pajak"]},
        {"name": "UU PPh", "keywords": ["pajak penghasilan", "PPh"]},
        {"name": "UU PPN", "keywords": ["pajak pertambahan nilai", "PPN"]},
    ]

    print("\nChecking for key tax regulations...")

    for reg in expected_regulations:
        print(f"\n🔍 Searching for {reg['name']}:")
        found_docs = []

        for doc in dataset:
            title = doc.get('regulation_title', '').lower()
            content = doc.get('content', '').lower()
            combined = title + " " + content

            matches = sum(1 for kw in reg['keywords'] if kw in combined)
            if matches > 0:
                found_docs.append({
                    'regulation_type': doc.get('regulation_type', 'Unknown'),
                    'regulation_number': doc.get('regulation_number', 'N/A'),
                    'year': doc.get('year', 'Unknown'),
                    'title': doc.get('regulation_title', 'N/A'),
                    'matches': matches
                })

        if found_docs:
            found_docs.sort(key=lambda x: x['matches'], reverse=True)
            print(f"   ✓ Found {len(found_docs)} potentially matching documents:")
            for doc in found_docs[:5]:
                print(f"      - {doc['regulation_type']} No. {doc['regulation_number']}/{doc['year']}")
                print(f"        {doc['title'][:70]}...")
        else:
            print(f"   ✗ NOT FOUND - This may explain poor RAG results!")


def main():
    """Run dataset coverage check"""
    print("\n" + "="*80)
    print("DATASET COVERAGE CHECK FOR TAX LAW QUERIES")
    print("="*80)

    dataset = load_dataset()
    if not dataset:
        print("\n✗ Cannot proceed without dataset")
        return 1

    # Run analyses
    highly_relevant = analyze_tax_law_coverage(dataset)
    check_specific_regulations(dataset)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & DIAGNOSIS")
    print("="*80)

    if not highly_relevant:
        print("\n❌ ROOT CAUSE IDENTIFIED:")
        print("   The dataset does NOT contain sufficient tax law documents!")
        print("\n   Why RAG returns irrelevant results:")
        print("   - Query asks about tax objections (keberatan pajak)")
        print("   - Dataset has no/few documents about tax law")
        print("   - RAG returns 'best match' from irrelevant documents")
        print("   - Results: Banking regulations, land use laws, etc.")
        print("\n   SOLUTION:")
        print("   1. Add UU No. 28 Tahun 2007 (UU KUP) to dataset")
        print("   2. Add UU No. 14 Tahun 2002 (UU Pengadilan Pajak)")
        print("   3. Add related tax regulations (PPh, PPN, etc.)")

    elif len(highly_relevant) < 10:
        print("\n⚠️  PARTIAL COVERAGE:")
        print(f"   Only {len(highly_relevant)} tax documents found")
        print("   This may lead to incomplete or poor quality answers")
        print("\n   RECOMMENDATION:")
        print("   Expand dataset with more tax law documents")

    else:
        print(f"\n✅ Good Coverage: {len(highly_relevant)} tax documents found")
        print("   If RAG still returns poor results, the issue is likely:")
        print("   - Semantic embedding quality")
        print("   - TF-IDF not tuned for Indonesian legal text")
        print("   - Weight configuration issues")
        print("   - Reranking model problems")

    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
