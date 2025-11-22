"""
Legal Form Generator

Generates legal document forms and templates based on user requirements.

File: core/form_generator.py
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from logger_utils import get_logger


class FormGenerator:
    """
    Generates legal forms and document templates
    """

    def __init__(self):
        self.logger = get_logger("FormGenerator")
        self._templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load form templates"""
        return {
            "surat_kuasa": {
                "name": "Surat Kuasa",
                "description": "Power of Attorney",
                "fields": [
                    {"name": "pemberi_kuasa", "label": "Pemberi Kuasa", "type": "text", "required": True},
                    {"name": "penerima_kuasa", "label": "Penerima Kuasa", "type": "text", "required": True},
                    {"name": "keperluan", "label": "Keperluan", "type": "textarea", "required": True},
                    {"name": "tanggal", "label": "Tanggal", "type": "date", "required": True},
                    {"name": "tempat", "label": "Tempat", "type": "text", "required": True},
                ]
            },
            "surat_pernyataan": {
                "name": "Surat Pernyataan",
                "description": "Statement Letter",
                "fields": [
                    {"name": "nama", "label": "Nama Lengkap", "type": "text", "required": True},
                    {"name": "tempat_lahir", "label": "Tempat Lahir", "type": "text", "required": True},
                    {"name": "tanggal_lahir", "label": "Tanggal Lahir", "type": "date", "required": True},
                    {"name": "alamat", "label": "Alamat", "type": "textarea", "required": True},
                    {"name": "nik", "label": "NIK", "type": "text", "required": True},
                    {"name": "isi_pernyataan", "label": "Isi Pernyataan", "type": "textarea", "required": True},
                ]
            },
            "perjanjian_kerja": {
                "name": "Perjanjian Kerja",
                "description": "Employment Agreement",
                "fields": [
                    {"name": "nama_perusahaan", "label": "Nama Perusahaan", "type": "text", "required": True},
                    {"name": "nama_karyawan", "label": "Nama Karyawan", "type": "text", "required": True},
                    {"name": "jabatan", "label": "Jabatan", "type": "text", "required": True},
                    {"name": "gaji", "label": "Gaji Pokok", "type": "number", "required": True},
                    {"name": "tanggal_mulai", "label": "Tanggal Mulai Kerja", "type": "date", "required": True},
                    {"name": "durasi", "label": "Durasi (bulan)", "type": "number", "required": False},
                ]
            },
            "pengaduan": {
                "name": "Surat Pengaduan",
                "description": "Complaint Letter",
                "fields": [
                    {"name": "kepada", "label": "Kepada", "type": "text", "required": True},
                    {"name": "dari", "label": "Dari", "type": "text", "required": True},
                    {"name": "perihal", "label": "Perihal", "type": "text", "required": True},
                    {"name": "isi_pengaduan", "label": "Isi Pengaduan", "type": "textarea", "required": True},
                    {"name": "lampiran", "label": "Lampiran", "type": "text", "required": False},
                ]
            },
            "somasi": {
                "name": "Surat Somasi",
                "description": "Warning Letter",
                "fields": [
                    {"name": "kepada", "label": "Kepada", "type": "text", "required": True},
                    {"name": "dari", "label": "Dari", "type": "text", "required": True},
                    {"name": "perihal", "label": "Perihal", "type": "text", "required": True},
                    {"name": "kronologi", "label": "Kronologi", "type": "textarea", "required": True},
                    {"name": "tuntutan", "label": "Tuntutan", "type": "textarea", "required": True},
                    {"name": "batas_waktu", "label": "Batas Waktu (hari)", "type": "number", "required": True},
                ]
            }
        }

    def list_templates(self) -> List[Dict[str, Any]]:
        """List available form templates"""
        templates = []
        for key, template in self._templates.items():
            templates.append({
                "id": key,
                "name": template["name"],
                "description": template["description"],
                "field_count": len(template["fields"])
            })
        return templates

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template details including fields"""
        if template_id not in self._templates:
            return None
        return self._templates[template_id]

    def generate_form(
        self,
        template_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a filled form document

        Args:
            template_id: Template identifier
            data: Form field data

        Returns:
            Generated form result
        """
        if template_id not in self._templates:
            return {
                "success": False,
                "error": f"Template not found: {template_id}"
            }

        template = self._templates[template_id]

        # Validate required fields
        missing = []
        for field in template["fields"]:
            if field["required"] and field["name"] not in data:
                missing.append(field["label"])

        if missing:
            return {
                "success": False,
                "error": f"Missing required fields: {', '.join(missing)}"
            }

        # Generate document based on template
        try:
            if template_id == "surat_kuasa":
                content = self._generate_surat_kuasa(data)
            elif template_id == "surat_pernyataan":
                content = self._generate_surat_pernyataan(data)
            elif template_id == "perjanjian_kerja":
                content = self._generate_perjanjian_kerja(data)
            elif template_id == "pengaduan":
                content = self._generate_pengaduan(data)
            elif template_id == "somasi":
                content = self._generate_somasi(data)
            else:
                content = self._generate_generic(template, data)

            return {
                "success": True,
                "template_name": template["name"],
                "content": content,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Form generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_surat_kuasa(self, data: Dict[str, Any]) -> str:
        """Generate Surat Kuasa"""
        return f"""
SURAT KUASA

Yang bertanda tangan di bawah ini:

Nama    : {data.get('pemberi_kuasa', '')}
(selanjutnya disebut "PEMBERI KUASA")

Dengan ini memberikan kuasa kepada:

Nama    : {data.get('penerima_kuasa', '')}
(selanjutnya disebut "PENERIMA KUASA")

----------- KHUSUS -----------

Untuk dan atas nama Pemberi Kuasa melakukan hal-hal sebagai berikut:

{data.get('keperluan', '')}

Demikian surat kuasa ini dibuat untuk dipergunakan sebagaimana mestinya.

{data.get('tempat', '')}, {data.get('tanggal', '')}

Pemberi Kuasa,                           Penerima Kuasa,



({data.get('pemberi_kuasa', '')})        ({data.get('penerima_kuasa', '')})
""".strip()

    def _generate_surat_pernyataan(self, data: Dict[str, Any]) -> str:
        """Generate Surat Pernyataan"""
        return f"""
SURAT PERNYATAAN

Yang bertanda tangan di bawah ini:

Nama                : {data.get('nama', '')}
Tempat/Tgl Lahir    : {data.get('tempat_lahir', '')}, {data.get('tanggal_lahir', '')}
NIK                 : {data.get('nik', '')}
Alamat              : {data.get('alamat', '')}

Dengan ini menyatakan dengan sebenar-benarnya bahwa:

{data.get('isi_pernyataan', '')}

Demikian surat pernyataan ini saya buat dengan penuh kesadaran dan tanpa ada paksaan dari pihak manapun.

{datetime.now().strftime('%d %B %Y')}
Yang membuat pernyataan,



({data.get('nama', '')})
""".strip()

    def _generate_perjanjian_kerja(self, data: Dict[str, Any]) -> str:
        """Generate Perjanjian Kerja"""
        return f"""
PERJANJIAN KERJA WAKTU TERTENTU

Pada hari ini, {data.get('tanggal_mulai', '')}, telah disepakati Perjanjian Kerja antara:

1. {data.get('nama_perusahaan', '')}
   (selanjutnya disebut "PIHAK PERTAMA" / "Perusahaan")

2. {data.get('nama_karyawan', '')}
   (selanjutnya disebut "PIHAK KEDUA" / "Karyawan")

Kedua belah pihak sepakat untuk mengadakan Perjanjian Kerja dengan ketentuan sebagai berikut:

Pasal 1 - JABATAN DAN TUGAS
Pihak Kedua dipekerjakan sebagai {data.get('jabatan', '')} dengan tugas dan tanggung jawab sesuai deskripsi jabatan.

Pasal 2 - JANGKA WAKTU
Perjanjian ini berlaku sejak {data.get('tanggal_mulai', '')} {"untuk jangka waktu " + str(data.get('durasi', '')) + " bulan" if data.get('durasi') else "untuk waktu tidak tertentu"}.

Pasal 3 - GAJI DAN TUNJANGAN
Pihak Kedua berhak menerima gaji pokok sebesar Rp {data.get('gaji', '')} per bulan.

Pasal 4 - HAK DAN KEWAJIBAN
Kedua belah pihak wajib mentaati peraturan perusahaan dan peraturan perundang-undangan yang berlaku.

Demikian perjanjian ini dibuat dan ditandatangani oleh kedua belah pihak.

PIHAK PERTAMA,                          PIHAK KEDUA,



({data.get('nama_perusahaan', '')})      ({data.get('nama_karyawan', '')})
""".strip()

    def _generate_pengaduan(self, data: Dict[str, Any]) -> str:
        """Generate Surat Pengaduan"""
        return f"""
SURAT PENGADUAN

Kepada Yth.
{data.get('kepada', '')}
di tempat

Perihal: {data.get('perihal', '')}

Dengan hormat,

Yang bertanda tangan di bawah ini:
Nama: {data.get('dari', '')}

Dengan ini mengajukan pengaduan sebagai berikut:

{data.get('isi_pengaduan', '')}

{f"Lampiran: {data.get('lampiran', '')}" if data.get('lampiran') else ""}

Demikian surat pengaduan ini kami sampaikan. Atas perhatian dan tindak lanjutnya, kami ucapkan terima kasih.

{datetime.now().strftime('%d %B %Y')}
Hormat kami,



({data.get('dari', '')})
""".strip()

    def _generate_somasi(self, data: Dict[str, Any]) -> str:
        """Generate Surat Somasi"""
        return f"""
SURAT SOMASI

Kepada Yth.
{data.get('kepada', '')}
di tempat

Perihal: SOMASI - {data.get('perihal', '')}

Dengan hormat,

Kami yang bertanda tangan di bawah ini:
{data.get('dari', '')}

Melalui surat ini menyampaikan hal-hal sebagai berikut:

KRONOLOGI KEJADIAN:
{data.get('kronologi', '')}

Berdasarkan hal tersebut di atas, dengan ini kami MENUNTUT:
{data.get('tuntutan', '')}

Kami memberikan waktu selama {data.get('batas_waktu', '')} hari sejak tanggal surat ini untuk memenuhi tuntutan tersebut. Apabila dalam jangka waktu yang ditentukan tidak ada tanggapan atau penyelesaian, maka kami akan menempuh jalur hukum.

Demikian surat somasi ini kami sampaikan, untuk dapat ditindaklanjuti sebagaimana mestinya.

{datetime.now().strftime('%d %B %Y')}



({data.get('dari', '')})
""".strip()

    def _generate_generic(
        self,
        template: Dict[str, Any],
        data: Dict[str, Any]
    ) -> str:
        """Generate generic form from template"""
        lines = [f"=== {template['name'].upper()} ===\n"]

        for field in template["fields"]:
            name = field["name"]
            label = field["label"]
            value = data.get(name, "")
            lines.append(f"{label}: {value}")

        lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        return "\n".join(lines)


# Singleton instance
_form_generator = None


def get_form_generator() -> FormGenerator:
    """Get or create form generator singleton"""
    global _form_generator
    if _form_generator is None:
        _form_generator = FormGenerator()
    return _form_generator


if __name__ == "__main__":
    print("=" * 60)
    print("FORM GENERATOR TEST")
    print("=" * 60)

    generator = FormGenerator()

    # List templates
    print("\nAvailable Templates:")
    templates = generator.list_templates()
    for t in templates:
        print(f"  - {t['id']}: {t['name']} ({t['description']})")

    # Test Surat Kuasa
    print("\n" + "-" * 60)
    print("TEST: Surat Kuasa")
    print("-" * 60)

    result = generator.generate_form("surat_kuasa", {
        "pemberi_kuasa": "John Doe",
        "penerima_kuasa": "Jane Smith",
        "keperluan": "Mengurus dokumen perizinan usaha",
        "tanggal": "22 November 2025",
        "tempat": "Jakarta"
    })

    if result['success']:
        print(f"\n✓ Generated: {result['template_name']}")
        print(f"\n{result['content']}")
    else:
        print(f"\n✗ Failed: {result['error']}")

    # Test Perjanjian Kerja
    print("\n" + "-" * 60)
    print("TEST: Perjanjian Kerja")
    print("-" * 60)

    result = generator.generate_form("perjanjian_kerja", {
        "nama_perusahaan": "PT Teknologi Indonesia",
        "nama_karyawan": "Ahmad Wijaya",
        "jabatan": "Software Engineer",
        "gaji": "15.000.000",
        "tanggal_mulai": "1 Januari 2025",
        "durasi": 12
    })

    if result['success']:
        print(f"\n✓ Generated: {result['template_name']}")
        print(f"\n{result['content'][:500]}...")
    else:
        print(f"\n✗ Failed: {result['error']}")

    # Test missing fields
    print("\n" + "-" * 60)
    print("TEST: Missing Fields (should fail)")
    print("-" * 60)

    result = generator.generate_form("surat_kuasa", {
        "pemberi_kuasa": "Test"
        # Missing required fields
    })

    if result['success']:
        print(f"\n✗ Should have failed")
    else:
        print(f"\n✓ Correctly failed: {result['error']}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
