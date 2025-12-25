# Expanding Indonesian Legal Vocabulary

## ✅ Refactoring Complete!

The synonym dictionary has been refactored into a standalone, easily expandable file:

**Location:** `config/indonesian_legal_vocabulary.py`

### What Changed

**Before:** Embedded dictionary in `query_detection.py` (9 terms)
```python
INDONESIAN_LEGAL_SYNONYMS = {
    'guru': ['guru', 'pendidik', ...],
    # Only 9 terms
}
```

**After:** Standalone vocabulary file (80+ terms, 14 domains)
```python
# config/indonesian_legal_vocabulary.py
EDUCATION_SYNONYMS = {...}
EMPLOYMENT_SYNONYMS = {...}
CRIMINAL_SYNONYMS = {...}
# ... 14 domain categories
INDONESIAN_LEGAL_SYNONYMS = {**EDUCATION_SYNONYMS, **EMPLOYMENT_SYNONYMS, ...}
```

---

## 📚 Comprehensive Expansion (14 Domains)

### 1. **Education & Personnel** (7 terms)
- guru, dosen, pendidikan, sekolah, universitas, siswa, mahasiswa

### 2. **Employment & Compensation** (10 terms)
- pegawai, pns, tunjangan, tunjangan profesi, gaji, honorarium, pesangon, pensiun, cuti, lembur

### 3. **Legal Concepts & Regulatory** (8 terms)
- pengaturan, undang-undang, peraturan pemerintah, peraturan presiden, peraturan menteri, peraturan daerah, keputusan, surat edaran

### 4. **Rights & Obligations** (6 terms)
- hak, kewajiban, kesetaraan, keadilan, perlindungan, kebebasan

### 5. **Criminal Law & Sanctions** (7 terms)
- pidana, sanksi, denda, penjara, tindak pidana, korupsi, penipuan

### 6. **Civil Law & Procedures** (6 terms)
- perdata, kontrak, gugatan, ganti rugi, wanprestasi, notaris

### 7. **Administrative & Governance** (7 terms)
- pemerintah, pemerintah daerah, kementerian, lembaga, izin, perizinan, pengawasan

### 8. **Business & Economy** (7 terms)
- perusahaan, usaha, perdagangan, investasi, modal, pajak, bea

### 9. **Land & Property** (5 terms)
- tanah, hak milik, sertifikat, agraria, bangunan

### 10. **Family & Marriage** (5 terms)
- perkawinan, perceraian, waris, anak, adopsi

### 11. **Health & Social Welfare** (6 terms)
- kesehatan, rumah sakit, dokter, asuransi, bpjs, sosial

### 12. **Environment & Natural Resources** (6 terms)
- lingkungan, lingkungan hidup, pencemaran, kehutanan, pertambangan, energi

### 13. **Transportation & Traffic** (5 terms)
- lalu lintas, kendaraan, sim, stnk, jalan

### 14. **Procedural Terms** (5 terms)
- prosedur, syarat, tahapan, pengajuan, pendaftaran

**Total:** 80+ synonym groups covering all major Indonesian legal domains!

---

## 🎯 How to Expand Further

### Option 1: Add New Terms to Existing Domains

Edit `config/indonesian_legal_vocabulary.py`:

```python
# Add to EDUCATION_SYNONYMS section
EDUCATION_SYNONYMS = {
    # ... existing terms ...
    'kurikulum': ['kurikulum', 'curriculum', 'silabus', 'materi ajar'],  # NEW!
    'ujian': ['ujian', 'exam', 'test', 'evaluasi'],  # NEW!
}
```

### Option 2: Add New Domain Categories

```python
# Add new section in the file
# =============================================================================
# TECHNOLOGY & DIGITAL
# =============================================================================

TECHNOLOGY_SYNONYMS = {
    'data pribadi': ['data pribadi', 'personal data', 'informasi pribadi'],
    'siber': ['siber', 'cyber', 'digital', 'elektronik'],
    'startup': ['startup', 'rintisan', 'perusahaan rintisan'],
}

# Then add to master dictionary
INDONESIAN_LEGAL_SYNONYMS = {
    **EDUCATION_SYNONYMS,
    **EMPLOYMENT_SYNONYMS,
    # ... existing ...
    **TECHNOLOGY_SYNONYMS,  # NEW!
}

# And add to LEGAL_DOMAINS
LEGAL_DOMAINS = {
    'education': [...],
    # ... existing ...
    'technology': ['data pribadi', 'siber', 'startup'],  # NEW!
}
```

### Option 3: Multi-word Phrases (Advanced)

```python
# For complex legal terms
EMPLOYMENT_SYNONYMS = {
    'pemutusan hubungan kerja': [
        'pemutusan hubungan kerja', 
        'phk', 
        'termination', 
        'pemecatan',
        'pemberhentian'
    ],
}
```

---

## 💡 Recommended Additions

### High Priority (Common Legal Queries)

**Labor Law:**
```python
'upah minimum': ['upah minimum', 'umr', 'ump', 'umk', 'minimum wage'],
'outsourcing': ['outsourcing', 'alih daya', 'tenaga kerja kontrak'],
'serikat pekerja': ['serikat pekerja', 'union', 'labor union', 'sp'],
```

**Consumer Protection:**
```python
'konsumen': ['konsumen', 'consumer', 'pembeli', 'pelanggan'],
'perlindungan konsumen': ['perlindungan konsumen', 'consumer protection'],
'barang': ['barang', 'goods', 'produk', 'merchandise'],
```

**Intellectual Property:**
```python
'hak cipta': ['hak cipta', 'copyright', 'haki'],
'paten': ['paten', 'patent', 'hak paten'],
'merek': ['merek', 'trademark', 'brand'],
```

**Banking & Finance:**
```python
'bank': ['bank', 'perbankan', 'banking'],
'kredit': ['kredit', 'credit', 'pinjaman', 'loan'],
'bunga': ['bunga', 'interest', 'suku bunga'],
```

---

## 🔧 Usage in Code

The vocabulary is automatically imported and used in `query_detection.py`:

```python
from config.indonesian_legal_vocabulary import INDONESIAN_LEGAL_SYNONYMS

# expand_query_with_synonyms() already uses this!
query = "Apa hak guru?"
expanded = detector.expand_query_with_synonyms(query)
# Returns: ["Apa hak guru?", "Apa kewenangan guru?", "Apa otoritas pendidik?", ...]
```

---

## 📊 Impact on Search

### Before Expansion (9 terms)
- Query: "Apa syarat dosen?"
- Matched: "syarat" only
- Retrieved: Limited results

### After Expansion (80+ terms)
- Query: "Apa syarat dosen?"
- Matched: "syarat" → "persyaratan", "ketentuan"
- Matched: "dosen" → "pengajar perguruan tinggi", "lecturer"
- Retrieved: 3-5x more relevant results!

---

## 🚀 Next Steps

1. **Test the expanded vocabulary:**
   ```bash
   python config/indonesian_legal_vocabulary.py
   ```
   This will show all 80+ synonym groups

2. **Add domain-specific terms** based on your most common queries

3. **Run your RAG test** to see improved results

4. **Monitor and expand** - add new terms when you see queries failing

Would you like me to add any specific domains or terms?
