"""
Indonesian Legal Vocabulary - Comprehensive Synonym Dictionary
Purpose: Improve RAG retrieval by expanding queries with legal domain synonyms
Maintainer: Easily expandable - add new terms and domains as needed
"""

# =============================================================================
# EDUCATION & PERSONNEL
# =============================================================================

EDUCATION_SYNONYMS = {
    'guru': ['guru', 'pendidik', 'tenaga pendidik', 'pengajar', 'educator'],
    'dosen': ['dosen', 'pengajar perguruan tinggi', 'tenaga pengajar', 'lecturer'],
    'pendidikan': ['pendidikan', 'pengajaran', 'pembelajaran', 'edukasi', 'pendidikan formal'],
    'sekolah': ['sekolah', 'lembaga pendidikan', 'institusi pendidikan', 'satuan pendidikan'],
    'universitas': ['universitas', 'perguruan tinggi', 'kampus', 'institusi tinggi'],
    'siswa': ['siswa', 'murid', 'peserta didik', 'pelajar'],
    'mahasiswa': ['mahasiswa', 'peserta didik tinggi', 'student'],
}

# =============================================================================
# EMPLOYMENT & COMPENSATION
# =============================================================================

EMPLOYMENT_SYNONYMS = {
    'pegawai': ['pegawai', 'karyawan', 'pekerja', 'tenaga kerja', 'employee'],
    'pns': ['pns', 'pegawai negeri sipil', 'asn', 'aparatur sipil negara'],
    'tunjangan': ['tunjangan', 'insentif', 'benefit', 'bantuan', 'allowance'],
    'tunjangan profesi': ['tunjangan profesi', 'tunjangan profesional', 'tunjangan pendidik', 'insentif profesi'],
    'gaji': ['gaji', 'upah', 'penghasilan', 'remunerasi', 'salary'],
    'honorarium': ['honorarium', 'fee', 'imbalan', 'upah jasa'],
    'pesangon': ['pesangon', 'uang penghargaan', 'kompensasi', 'severance'],
    'pensiun': ['pensiun', 'dana pensiun', 'hari tua', 'retirement'],
    'cuti': ['cuti', 'izin', 'leave', 'waktu istirahat'],
    'lembur': ['lembur', 'kerja lembur', 'overtime', 'kerja di luar jam'],
}

# =============================================================================
# LEGAL CONCEPTS & REGULATORY
# =============================================================================

REGULATORY_SYNONYMS = {
    'pengaturan': ['pengaturan', 'peraturan', 'ketentuan', 'aturan', 'regulasi', 'regulation'],
    'undang-undang': ['undang-undang', 'uu', 'statute', 'law'],
    'peraturan pemerintah': ['peraturan pemerintah', 'pp', 'government regulation'],
    'peraturan presiden': ['peraturan presiden', 'perpres', 'presidential regulation'],
    'peraturan menteri': ['peraturan menteri', 'permen', 'ministerial regulation'],
    'peraturan daerah': ['peraturan daerah', 'perda', 'local regulation'],
    'keputusan': ['keputusan', 'keppres', 'kepmen', 'decision', 'decree'],
    'surat edaran': ['surat edaran', 'se', 'circular letter'],
}

# =============================================================================
# RIGHTS & OBLIGATIONS
# =============================================================================

RIGHTS_SYNONYMS = {
    'hak': ['hak', 'kewenangan', 'otoritas', 'rights'],
    'kewajiban': ['kewajiban', 'tanggung jawab', 'obligation', 'duty'],
    'kesetaraan': ['kesetaraan', 'kesamaan', 'kesejajaran', 'equality'],
    'keadilan': ['keadilan', 'justice', 'fairness', 'equity'],
    'perlindungan': ['perlindungan', 'protection', 'jaminan', 'safeguard'],
    'kebebasan': ['kebebasan', 'freedom', 'liberty', 'kemerdekaan'],
}

# =============================================================================
# CRIMINAL LAW & SANCTIONS
# =============================================================================

CRIMINAL_SYNONYMS = {
    'pidana': ['pidana', 'criminal', 'hukum pidana', 'penal'],
    'sanksi': ['sanksi', 'hukuman', 'penalty', 'punishment'],
    'denda': ['denda', 'fine', 'multa', 'uang penalty'],
    'penjara': ['penjara', 'kurungan', 'imprisonment', 'tahanan'],
    'tindak pidana': ['tindak pidana', 'kejahatan', 'crime', 'criminal act'],
    'korupsi': ['korupsi', 'corruption', 'gratifikasi', 'suap'],
    'penipuan': ['penipuan', 'fraud', 'tipu muslihat', 'deception'],
}

# =============================================================================
# CIVIL LAW & PROCEDURES
# =============================================================================

CIVIL_SYNONYMS = {
    'perdata': ['perdata', 'civil', 'hukum perdata'],
    'kontrak': ['kontrak', 'perjanjian', 'contract', 'agreement'],
    'gugatan': ['gugatan', 'lawsuit', 'tuntutan', 'claim'],
    'ganti rugi': ['ganti rugi', 'kompensasi', 'indemnity', 'damages'],
    'wanprestasi': ['wanprestasi', 'cidera janji', 'breach of contract'],
    'notaris': ['notaris', 'notary', 'pejabat pembuat akta'],
}

# =============================================================================
# ADMINISTRATIVE & GOVERNANCE
# =============================================================================

ADMINISTRATIVE_SYNONYMS = {
    'pemerintah': ['pemerintah', 'government', 'negara', 'state'],
    'pemerintah daerah': ['pemerintah daerah', 'pemda', 'local government'],
    'kementerian': ['kementerian', 'ministry', 'departemen'],
    'lembaga': ['lembaga', 'badan', 'agency', 'institution'],
    'izin': ['izin', 'permission', 'permit', 'lisensi'],
    'perizinan': ['perizinan', 'licensing', 'permitting'],
    'pengawasan': ['pengawasan', 'supervision', 'kontrol', 'monitoring'],
}

# =============================================================================
# BUSINESS & ECONOMY
# =============================================================================

BUSINESS_SYNONYMS = {
    'perusahaan': ['perusahaan', 'company', 'korporasi', 'corporation'],
    'usaha': ['usaha', 'business', 'bisnis', 'enterprise'],
    'perdagangan': ['perdagangan', 'trade', 'dagang', 'commerce'],
    'investasi': ['investasi', 'investment', 'penanaman modal'],
    'modal': ['modal', 'capital', 'equity'],
    'pajak': ['pajak', 'tax', 'pungutan', 'levy'],
    'bea': ['bea', 'customs', 'tarif', 'duty'],
}

# =============================================================================
# LAND & PROPERTY
# =============================================================================

PROPERTY_SYNONYMS = {
    'tanah': ['tanah', 'land', 'lahan', 'properti'],
    'hak milik': ['hak milik', 'ownership', 'kepemilikan'],
    'sertifikat': ['sertifikat', 'certificate', 'bukti hak'],
    'agraria': ['agraria', 'agrarian', 'pertanahan'],
    'bangunan': ['bangunan', 'building', 'gedung', 'konstruksi'],
}

# =============================================================================
# FAMILY & MARRIAGE
# =============================================================================

FAMILY_SYNONYMS = {
    'perkawinan': ['perkawinan', 'marriage', 'nikah', 'pernikahan'],
    'perceraian': ['perceraian', 'divorce', 'cerai', 'talak'],
    'waris': ['waris', 'inheritance', 'warisan', 'harta pusaka'],
    'anak': ['anak', 'child', 'keturunan', 'offspring'],
    'adopsi': ['adopsi', 'adoption', 'pengangkatan anak'],
}

# =============================================================================
# HEALTH & SOCIAL WELFARE
# =============================================================================

HEALTH_SYNONYMS = {
    'kesehatan': ['kesehatan', 'health', 'medis', 'medical'],
    'rumah sakit': ['rumah sakit', 'hospital', 'rs', 'faskes'],
    'dokter': ['dokter', 'doctor', 'physician', 'tenaga medis'],
    'asuransi': ['asuransi', 'insurance', 'jaminan', 'coverage'],
    'bpjs': ['bpjs', 'jaminan kesehatan', 'jkn', 'jamkesmas'],
    'sosial': ['sosial', 'social', 'kemasyarakatan', 'masyarakat'],
}

# =============================================================================
# ENVIRONMENT & NATURAL RESOURCES
# =============================================================================

ENVIRONMENT_SYNONYMS = {
    'lingkungan': ['lingkungan', 'environment', 'ekologi', 'alam'],
    'lingkungan hidup': ['lingkungan hidup', 'environmental', 'ekosistem'],
    'pencemaran': ['pencemaran', 'pollution', 'kontaminasi', 'polusi'],
    'kehutanan': ['kehutanan', 'forestry', 'hutan', 'forest'],
    'pertambangan': ['pertambangan', 'mining', 'tambang', 'galian'],
    'energi': ['energi', 'energy', 'daya', 'power'],
}

# =============================================================================
# TRANSPORTATION & TRAFFIC
# =============================================================================

TRANSPORTATION_SYNONYMS = {
    'lalu lintas': ['lalu lintas', 'traffic', 'transportasi'],
    'kendaraan': ['kendaraan', 'vehicle', 'mobil', 'motor'],
    'sim': ['sim', 'surat izin mengemudi', 'driving license'],
    'stnk': ['stnk', 'surat tanda nomor kendaraan', 'vehicle registration'],
    'jalan': ['jalan', 'road', 'raya', 'street'],
}

# =============================================================================
# PROCEDURAL TERMS
# =============================================================================

PROCEDURAL_SYNONYMS = {
    'prosedur': ['prosedur', 'procedure', 'tata cara', 'mekanisme'],
    'syarat': ['syarat', 'requirement', 'ketentuan', 'persyaratan'],
    'tahapan': ['tahapan', 'stages', 'fase', 'proses'],
    'pengajuan': ['pengajuan', 'submission', 'permohonan', 'application'],
    'pendaftaran': ['pendaftaran', 'registration', 'registrasi', 'enrollment'],
}

# =============================================================================
# COMBINED MASTER DICTIONARY
# =============================================================================

INDONESIAN_LEGAL_SYNONYMS = {
    **EDUCATION_SYNONYMS,
    **EMPLOYMENT_SYNONYMS,
    **REGULATORY_SYNONYMS,
    **RIGHTS_SYNONYMS,
    **CRIMINAL_SYNONYMS,
    **CIVIL_SYNONYMS,
    **ADMINISTRATIVE_SYNONYMS,
    **BUSINESS_SYNONYMS,
    **PROPERTY_SYNONYMS,
    **FAMILY_SYNONYMS,
    **HEALTH_SYNONYMS,
    **ENVIRONMENT_SYNONYMS,
    **TRANSPORTATION_SYNONYMS,
    **PROCEDURAL_SYNONYMS,
}

# =============================================================================
# DOMAIN CATEGORIES (for domain-specific search optimization)
# =============================================================================

LEGAL_DOMAINS = {
    'education': ['pendidikan', 'guru', 'dosen', 'sekolah', 'universitas'],
    'employment': ['pegawai', 'kerja', 'gaji', 'tunjangan', 'pns'],
    'criminal': ['pidana', 'sanksi', 'hukuman', 'kejahatan', 'korupsi'],
    'civil': ['perdata', 'kontrak', 'gugatan', 'ganti rugi'],
    'administrative': ['pemerintah', 'izin', 'perizinan', 'pengawasan'],
    'business': ['perusahaan', 'usaha', 'perdagangan', 'investasi'],
    'property': ['tanah', 'hak milik', 'agraria', 'bangunan'],
    'family': ['perkawinan', 'perceraian', 'waris', 'anak'],
    'health': ['kesehatan', 'rumah sakit', 'dokter', 'asuransi'],
    'environment': ['lingkungan', 'pencemaran', 'kehutanan', 'pertambangan'],
    'transportation': ['lalu lintas', 'kendaraan', 'sim', 'jalan'],
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("INDONESIAN LEGAL VOCABULARY - COMPREHENSIVE SYNONYM DICTIONARY")
    print("=" * 80)
    
    print(f"\nTotal synonym groups: {len(INDONESIAN_LEGAL_SYNONYMS)}")
    print(f"Total legal domains: {len(LEGAL_DOMAINS)}")
    
    print("\n--- Sample Synonyms ---")
    for term, synonyms in list(INDONESIAN_LEGAL_SYNONYMS.items())[:5]:
        print(f"\n'{term}' →")
        print(f"  {', '.join(synonyms)}")
    
    print("\n--- Legal Domains ---")
    for domain, keywords in LEGAL_DOMAINS.items():
        print(f"\n{domain.upper()}: {', '.join(keywords)}")
    
    print("\n" + "=" * 80)
    print("To add new terms: Edit the relevant section above")
    print("=" * 80)
