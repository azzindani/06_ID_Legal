# ✅ Legal Vocabulary - Renamed & Massively Expanded!

## New Name: `legal_vocab.py` (Much Shorter!)

**Old:** `config/indonesian_legal_vocabulary.py`  
**New:** `config/legal_vocab.py` ✅

---

## 📊 Expansion Complete!

### Before vs After

| Metric | Before | After | Increase |
|--------|--------|-------|----------|
| **Synonym Groups** | 9 | **150+** | 🔥 **16x more** |
| **Domains** | 1 | **23** | 🔥 **23x more** |
| **Coverage** | Basic | **Comprehensive** | ✅ All major legal fields |

---

## 🎯 23 Legal Domains Now Covered

1. **Education & Personnel** (11 terms) - guru, dosen, kurikulum, ujian, beasiswa...
2. **Employment & Labor** (16 terms) - PHK, outsourcing, upah minimum, serikat pekerja...
3. **Regulatory** (9 terms) - UU, PP, Perpres, Permen, Perda...
4. **Rights & Obligations** (8 terms) - HAM, diskriminasi, kesetaraan...
5. **Criminal Law** (12 terms) - korupsi, narkotika, terorisme, pencucian uang...
6. **Civil Law** (9 terms) - mediasi, arbitrase, somasi...
7. **Administrative** (8 terms) - TUN, perizinan, pengawasan...
8. **Business** (11 terms) - PT, CV, koperasi, UMKM, saham, dividen...
9. **Tax & Fiscal** (9 terms) - PPh, PPN, PBB, NPWP, SPT... 🆕
10. **Property & Land** (9 terms) - HGB, HGU, IMB, sertifikat...
11. **Family & Marriage** (8 terms) - nafkah, hak asuh, wasiat...
12. **Health & Welfare** (8 terms) - malpraktik, farmasi, BPJS...
13. **Consumer Protection** (8 terms) - garansi, recall, kedaluwarsa... 🆕
14. **Intellectual Property** (8 terms) - hak cipta, paten, merek, royalti... 🆕
15. **Banking & Finance** (9 terms) - kredit, deposito, fintech, KPR... 🆕
16. **Environment** (9 terms) - AMDAL, emisi, limbah...
17. **Transportation** (8 terms) - tilang, kecelakaan, parkir...
18. **Technology & Digital** (7 terms) - siber, privasi, e-commerce, hacker... 🆕
19. **Immigration** (8 terms) - KITAS, KITAP, WNI, WNA... 🆕
20. **Insurance** (6 terms) - premi, klaim, polis... 🆕
21. **Procedural** (8 terms) - verifikasi, validasi, pemeriksaan...
22. **Bankruptcy & Debt** (6 terms) - pailit, PKPU, kreditor, debitor... 🆕
23. **Competition Law** (5 terms) - monopoli, kartel, KPPU... 🆕

🆕 = **Newly added domains!**

---

## 💡 Key New Additions

### Labor Law (Highly Requested)
- `phk` → pemutusan hubungan kerja, termination, pemecatan
- `outsourcing` → alih daya, tenaga kerja kontrak
- `upah minimum` → UMR, UMP, UMK, minimum wage
- `serikat pekerja` → union, labor union, SP

### Tax (Critical for Business)
- `pph` → pajak penghasilan, income tax
- `ppn` → pajak pertambahan nilai, VAT
- `pbb` → pajak bumi dan bangunan, property tax
- `npwp` → nomor pokok wajib pajak, tax ID

### Technology & Digital (Modern Legal Issues)
- `data pribadi` → personal data, informasi pribadi
- `siber` → cyber, digital, elektronik
- `e-commerce` → perdagangan elektronik, online shop
- `ransomware` → malware, virus

### Consumer Protection
- `konsumen` → consumer, pembeli, pelanggan
- `garansi` → warranty, guarantee, jaminan
- `recall` → penarikan produk, product recall

### Banking & Finance
- `kredit` → credit, pinjaman, loan
- `fintech` → financial technology, teknologi finansial
- `kpr` → kredit pemilikan rumah, mortgage

---

## 🧪 Test It!

Run on your server:
```bash
python config/legal_vocab.py
```

**Expected output:**
```
================================================================================
INDONESIAN LEGAL VOCABULARY (legal_vocab.py)
================================================================================

📊 STATISTICS:
   Total synonym groups: 150+
   Total legal domains: 23
   Total synonym variants: 500+

📚 DOMAIN BREAKDOWN:
   • Employment & Labor: 16 terms
   • Criminal Law: 12 terms
   • Business: 11 terms
   ...
```

---

## ✅ Already Integrated

The new file is automatically used in:
- `core/search/query_detection.py` (import updated)
- `expand_query_with_synonyms()` method
- All RAG searches benefit immediately!

**No code changes needed - just works!** 🎉

---

## 🚀 Impact on Your Queries

### Example: Teacher Query
**Query:** "Apa hak guru untuk mendapat gaji?"

**Before (9 synonyms):**
- Expanded to: 2-3 variations

**After (150+ synonyms):**
- `guru` → guru, pendidik, tenaga pendidik, pengajar
- `hak` → hak, kewenangan, otoritas, rights
- `gaji` → gaji, upah, penghasilan, remunerasi, salary
- **Expanded to: 8-10 variations!**

### Example: Tax Query
**Query:** "Bagaimana cara bayar pajak penghasilan?"

**Before:** ❌ No tax synonyms
**After:** ✅ 
- `pajak penghasilan` → pph, income tax
- Result: Much better matching!

### Example: Digital Query
**Query:** "Aturan tentang data pribadi?"

**Before:** ❌ No tech synonyms
**After:** ✅
- `data pribadi` → personal data, informasi pribadi
- Result: Finds privacy regulations!

---

## 📝 How to Add More

Edit `config/legal_vocab.py`:

```python
# Find the relevant section, e.g., TECHNOLOGY_SYNONYMS
TECHNOLOGY_SYNONYMS = {
    # ... existing ...
    'blockchain': ['blockchain', 'distributed ledger', 'rantai blok'],  # ADD THIS!
    'nft': ['nft', 'non-fungible token', 'token digital'],  # ADD THIS!
}
```

Then add to master dict (already done automatically via `**TECHNOLOGY_SYNONYMS`).

That's it! Next query will use the new synonyms.

---

## 🎯 Recommended Next Additions

Based on common Indonesian legal queries:

1. **Construction Law** - izin bangunan, kontraktor, IMB details
2. **Public Procurement** - tender, lelang, pengadaan barang
3. **Maritime Law** - pelayaran, kapal, pelabuhan
4. **Energy Law** - listrik, PLN, energi terbarukan
5. **Education Accreditation** - akreditasi, BAN-PT, standar pendidikan

Want me to add any of these?
