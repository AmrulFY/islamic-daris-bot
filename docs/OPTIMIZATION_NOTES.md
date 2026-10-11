# Catatan Optimasi Latensi Muslim Bot

Status per 2026-10-11 (diperbarui setelah FTS5 + cap konteks: Langkah 9). Dokumen ini merangkum apa yang sudah dikerjakan untuk
mempercepat waktu dari prompt sampai jawaban, hasil pengukurannya, dan target
perbaikan berikutnya.

## 1. Masalah awal

Fast-path deterministik dan cache jawaban sudah ada, tetapi hanya menangkap
kasus sempit (lookup bernomor seperti "2:255", dan pertanyaan yang sama persis).
Pertanyaan topikal ("ayat tentang sabar") tetap masuk ke agent loop dan
memakan 20-100 detik.

## 2. Temuan dari pengukuran

Instrumentasi per request ditambahkan (baris log `timings`). Hasilnya:

| Temuan | Bukti |
|---|---|
| Tool MCP praktis gratis | 0.01-0.2 dtk per panggilan, total di bawah 2% dari waktu |
| Waktu habis di putaran LLM | 2-10 panggilan LLM per request, masing-masing 4-6 dtk atau lebih |
| Model mengabaikan batas "2-3 tool call" | 10 putaran pada "hadits senyum sedekah" (61.6 dtk) |
| Mengganti model tidak mengubah pola | Model kedua: median 45.4 dtk, terlama 97.5 dtk |
| `search_hadith` di MCP selalu kosong | Tabel `chapters` kosong dan indeks FTS kosong di database hadits, sehingga `JOIN chapters` di query LIKE tidak pernah menghasilkan baris. `total` dihitung tanpa join itu, jadi tampak ada hasil (`data: null`, `total: 226`). Ini penyebab utama agent mengulang pencarian 7 kali |

## 3. Yang sudah dikerjakan

### Langkah 0: pengukuran
- Class `RunTimings` di `muslim_bot/agent.py`. Setiap request menulis satu baris
  log: `total`, `mcp_setup`, `retrieval`, `ttft`, jumlah dan lama panggilan LLM,
  jumlah dan lama tiap tool, `other`.
- Catatan: `other` kemungkinan besar juga waktu LLM (waktu tunggu gateway
  sebelum event pertama tidak masuk hitungan `llm_time`). Ini dugaan, belum
  dibuktikan.

### Langkah 1: overhead tetap
- `cache_tools_list=True` pada server MCP (tidak ada `list_tools` per run).
- Health check koneksi MCP dibatasi interval (`MCP_HEALTH_CHECK_INTERVAL_SECONDS`,
  default 30 dtk), dan dipaksa ulang setelah run gagal. Hasil terukur:
  `mcp_setup=0.00s`.
- `parallel_tool_calls=True` (`AGENT_PARALLEL_TOOL_CALLS`). Terbukti bekerja:
  dua tool call terjadi pada detik yang sama.

### Langkah 2: pipeline retrieve-then-answer
- `muslim_bot/retrieval.py`: kode mengambil kata kunci (stopword Indonesia, varian
  tanpa imbuhan), mencari langsung di SQLite lokal (Quran dan hadits), 10-750 ms,
  tanpa LLM. Kata pendek hanya dicocokkan di awal kata ("riba" tidak cocok dengan
  "beribadah"). Rantai perawi hadits dibuang (`hadith_matn`).
- Satu panggilan LLM tanpa tool menyusun jawaban dari dalil yang ditemukan, dengan
  streaming (`PIPELINE_INSTRUCTIONS`, `_answer_from_context` di `agent.py`).
- Permintaan struktural (surah, juz, kitab, bab, daftar) tetap memakai agent loop
  lama. Jika retrieval error, otomatis kembali ke agent loop.
- Rollback: `AGENT_PIPELINE_ENABLED=false` lalu restart.

### Langkah 3: pencarian semantik sebagai cadangan
- `muslim_bot/semantic.py` dan `muslim_bot/semantic_index.py`. Embedding memakai
  `text-embedding-3-small` (512 dimensi) lewat gateway LLM yang sama. Cosine
  similarity dengan numpy di memori (sekitar 40 ribu vektor, sekitar 80 MB).
- Awalnya dipicu hanya untuk sumber yang lemah; sekarang selalu dijalankan
  paralel dengan pencarian kata dan digabung selang-seling (lihat Langkah 5).
- Indeks sudah dibangun: 6.236 ayat dan 34.186 hadits, disimpan di
  `.data/semantic.db`. Sekitar 4.900 hadits tanpa terjemahan Indonesia diindeks
  dari teks Inggris (model multibahasa). Dibangun ulang dengan
  `python -m muslim_bot.semantic_index` (bisa dilanjutkan jika terputus).
- Degradasi aman: indeks tidak ada, panggilan embedding gagal, atau lebih dari
  4 dtk berarti hasil semantik kosong, pipeline lanjut dengan hasil kata.
- Pemanasan saat startup (`warm_up`, dipanggil dari `lifespan` di `api.py`):
  panggilan embedding pertama pada koneksi dingin pernah memakan 16,6 dtk
  (selanjutnya 0,3-0,6 dtk), jauh di atas timeout, sehingga tanpa pemanasan
  request semantik pertama setelah restart gagal diam-diam.

### Langkah 4: eksperimen kualitas semantik
Dua ide diuji dengan data sebelum diterapkan. Dalil target dipilih oleh saya
(belum divalidasi pemilik produk), dan sampelnya kecil.

- **Embed query yang dibersihkan: ditolak.** Pada 7 pertanyaan, dalil target masuk
  8 teratas pada 3 kasus (query mentah), 2 kasus (pembersihan ringan: buang
  "ayat", "tentang", "dalil", dst.), dan 3 kasus (gabungan keduanya). Pembersihan
  penuh dengan stopword bahkan membuang "orang" dari "orang tua". Hasilnya
  seri atau lebih buruk, jadi query mentah tetap dipakai.
- **Perbanyak kandidat: diterapkan.** Pada 10 pertanyaan, dalil target masuk
  kandidat 4 teratas pada 20% kasus, 8 teratas 40%, 12 teratas 60%, lalu
  mendatar (16 dan 20 sama dengan 12). Kandidat semantik naik dari 4 ke 12 per
  sumber (`MAX_CONTEXT_PER_SOURCE` di `retrieval.py`), dengan batas total 12 per
  sumber di konteks. Pada uji "berbakti pada orang tua yang sudah renta",
  Al-Isra 17:23 kini ikut dikutip di jawaban (sebelumnya tidak masuk kandidat).
  Sampel latensi: total 20.7 dtk dengan 16 dalil di konteks, retrieval 4.0 dtk
  (termasuk pemuatan indeks pada request pertama).

### Langkah 5: evaluasi retrieval dan penggabungan hibrida
- Set evaluasi dari 20 dalil yang diberikan pemilik produk: `eval/retrieval_cases.json`
  dan skrip `eval/run_retrieval_eval.py` (`python -m eval.run_retrieval_eval`;
  tidak memanggil LLM, hanya embedding). Memakai jalur kode yang sama dengan
  produksi (`MuslimBotRunner.gather_passages`). Tiga gaya pertanyaan per kasus:
  `typed` ("Ayat/Hadits tentang X"), `neutral` ("Dalil tentang X", kedua sumber
  bersaing), `gist` (deskripsi isi dalil dari kolom "Inti relevansi").
- Catatan data: nomor Sahih Muslim di database berbeda dari penomoran umum.
  Item 14, 15, 19 dipetakan lewat isi (#6643/#6644, #6853, #283/#284). Item 11
  (Muslim 2999, sabar dan syukur) tidak ditemukan di database, dikeluarkan dari
  penilaian (19 kasus yang dinilai). Nomor Bukhari cocok persis.
- Temuan: strategi produksi sebelumnya (semantik hanya bila kata lemah) adalah
  yang terburuk untuk pertanyaan pendek. Pertanyaan berkata kunci tunggal
  ("sabar") selalu dianggap kuat oleh pencocokan kata, sehingga semantik tidak
  pernah dipicu dan konteks hanya berisi 4 ayat terpendek yang memuat kata itu.
- Perbandingan strategi (jumlah kasus dengan dalil target di 12 kandidat, dari 19):

| Strategi | typed | neutral | gist |
|---|---|---|---|
| Hanya kata | 2 | 2 | 9 |
| Hanya semantik | 4 | 4 | 9 |
| Semantik hanya bila kata lemah (sebelumnya) | 1 | 2 | 11 |
| **Selang-seling semantik + kata (sekarang)** | **4** | **6** | **12** |
| RRF | 4 | 6 | 12 |

- Diterapkan: semantik selalu jalan paralel dengan pencarian kata, hasil
  digabung selang-seling (`interleave_passages`), maksimal 12 per sumber.
  Retrieval total terukur 0,37-0,63 dtk pada kondisi hangat (8 pertanyaan).
- Dibuang: `weak_kinds` dan `merge_passages` (tidak dipakai lagi).
- Keputusan pemilik produk: tidak perlu mengeluarkan dalil yang persis sama
  dengan dalil target, karena satu topik bisa punya banyak dalil yang benar.
  Jadi angka evaluasi di atas adalah batas bawah yang ketat, bukan ukuran
  kualitas jawaban. Yang dijaga: dalil relevan, rujukan terverifikasi, dan model
  jujur saat tidak menemukan dalil.

### Langkah 6: perbaikan database hadits
Akar masalah `search_hadith` selalu kosong, ada tiga lapis di seeder Go
(`hadith-api-go-run`, `internal/services/seeder.go`):
1. Seeder membaca `section` / `section_detail` dari JSON edisi, padahal kuncinya
   `sections` / `section_details` (jamak). Peta bab selalu kosong, tabel
   `chapters` tidak pernah terisi.
2. Semua hadits ditulis dengan `chapter_id = 1`.
3. Trigger FTS dan query LIKE sama-sama `JOIN chapters`. Tanpa bab, indeks FTS
   kosong dan query LIKE tidak menghasilkan baris, sementara query hitungnya
   tidak memakai join itu (`data: null` dengan `total > 0`).

Perbaikan dilakukan di repo ini, tanpa mengubah repo Go:
- `muslim_bot/hadith_repair.py` (`python -m muslim_bot.hadith_repair`): mengunduh
  enam berkas edisi (sumber yang sama dengan seeder), membuat bab dari
  `sections`, memetakan tiap hadits ke bab lewat rentang `section_details`
  (hadits di luar rentang masuk bab "Lainnya"), lalu membangun ulang `hadith_fts`.
  Semua unduhan selesai dulu sebelum database disentuh, dan cadangan `.bak`
  dibuat otomatis. Aman dijalankan ulang.
- `bootstrap.py` memanggilnya otomatis (setelah seed, dan untuk database lama yang
  rusak), supaya deploy Docker yang memakai seeder Go yang sama tidak menghasilkan
  database rusak lagi. Jika unduhan gagal, startup tetap lanjut dengan peringatan.
- Diterapkan ke database lokal: 337 bab, 103.365 baris FTS, tidak ada hadits tanpa
  bab. Contoh pemetaan benar: Bukhari #1 ke bab "Revelation", #6018 ke "Good
  Manners and Form (Al-Adab)". `search_hadith` di MCP kini mengembalikan data
  ("niat": 80 hasil, "sedekah": 342). Ukuran file naik dari 92 MB ke 231 MB
  (indeks FTS menyimpan salinan teks). Cadangan lama: `hadith.db.bak`.
- Tidak berubah: id hadits (indeks semantik tetap valid) dan pipeline, yang
  membaca `hadith_texts` langsung.
- Masih berlaku: pencarian FTS memakai AND antar-kata dalam satu teks, dan
  tidak mencocokkan imbuhan (contoh: "senyum sedekah" tetap 0 hasil). Itu
  perilaku server Go, bukan bagian dari perbaikan ini.
- Seeder Go sendiri belum diperbaiki (repo terpisah). Perbaikan di sana: kunci
  JSON jamak, `chapter_id` dari rentang bab, dan urutan seed bab sebelum hadits.

### Langkah 7: pertanyaan lanjutan (follow-up)
Sebelumnya retrieval hanya memakai pesan terbaru, jadi "jelaskan ayat tadi" atau
"bagaimana dengan hadits?" tidak punya kata kunci dan konteks dalil kosong
(model diinstruksikan mengatakan "dalil tidak ditemukan").

- Pesan tanpa kata kunci sendiri meminjam topik dari pertanyaan sebelumnya
  (`build_retrieval_query` di `retrieval.py`). Jika pesan menyebut sumber
  ("hadits"/"ayat"), pesan itu yang menentukan sumbernya; jika tidak, sumber
  mengikuti pertanyaan sebelumnya.
- Dalil yang dikutip pada jawaban sebelumnya dibaca ulang dari database
  (`cited_passages`) dan dikirim sebagai "DALIL YANG DIBAHAS PADA JAWABAN
  SEBELUMNYA", sehingga penjelasan memakai teks asli, bukan ingatan model.
- Lanjutan berantai ("ayat tentang riba" -> "jelaskan ayat tadi" -> "bagaimana
  dengan hadits?") ditangani dengan menelusuri 4 giliran terakhir
  (`follow_up_sources`, `SessionStore.recent_turns`). Versi pertama hanya melihat
  satu giliran dan kehilangan topik pada lanjutan kedua; ketahuan lewat uji
  percakapan nyata.
- Prompt pipeline: salam, terima kasih, atau permintaan menyederhanakan jawaban
  tidak lagi dijawab dengan "pencarian database tidak menemukan dalil".
- Kegagalan panggilan ke gateway LLM (timeout, koneksi, status error) sebelumnya
  menjadi HTTP 500 mentah. Sekarang 504 / 503 / 502 di `/chat`, dan event `error`
  di `/chat/stream`.
- Uji nyata 4 giliran dalam satu sesi (riba -> jelaskan lebih sederhana ->
  bagaimana dengan hadits -> terima kasih): semua rujukan terverifikasi, tiap
  giliran 8-30 dtk.

### Langkah 8: kesiapan deploy indeks semantik
- Indeks `semantic.db` sebelumnya hanya ada di mesin lokal dan default ke
  `/app/.data` (bukan volume), sehingga di deploy fitur semantik mati diam-diam
  dan indeks hilang tiap redeploy.
- `SEMANTIC_DB_PATH=/app/data/semantic.db` di Dockerfile dan compose, plus
  variabel baru di compose (`SEMANTIC_*`, `EMBEDDING_*`, `AGENT_PIPELINE_ENABLED`,
  `AGENT_PARALLEL_TOOL_CALLS`, `MCP_HEALTH_CHECK_INTERVAL_SECONDS`).
- Pembangunan otomatis di latar belakang saat startup API
  (`_semantic_startup` di `api.py`, `SEMANTIC_AUTO_INDEX`, default true). API
  tetap melayani dengan pencarian kata selama indeks dibangun, dan pembangunan
  dilanjutkan dari titik terakhir jika proses restart. Berhenti di antara batch
  saat shutdown, jadi tidak menahan penghentian server.
- Indeks punya penanda "lengkap" per sumber (`meta` di `semantic.db`). Pencari
  hanya memuat sumber yang lengkap (tidak pernah indeks setengah jadi), mencoba
  lagi tiap 30 dtk selama belum lengkap, dan memuat ulang otomatis setelah build.
  Tidak lagi membuat berkas kosong jika indeks belum ada.
- `GET /health` punya field `semantic_index` (`ready`/`building`/`missing`/
  `disabled`). Hanya informasi, tidak pernah membuat health check gagal.
- Panduan VPS (`deploy/OPENCLAW_VPS.md`) menjelaskan build pertama (sekitar
  7-10 menit, beberapa sen), perbaikan database hadits saat boot, dan cara
  menyalin `semantic.db` yang sudah ada.
- Belum diuji: build Docker dan boot kontainer sebenarnya (Docker tidak
  dijalankan di sini). Yang diuji: logika build/resume/berhenti, pemuatan indeks,
  dan konfigurasi, lewat test.

## 4. Hasil

Sampel kecil (6 pertanyaan baru per kondisi, model yang sama saat dibandingkan).
Pertanyaannya tidak identik antar kondisi, jadi ini indikasi, bukan benchmark.

| | Agent loop | Pipeline |
|---|---|---|
| Median | 45.4 dtk | 17.2 dtk |
| Rata-rata | 56.2 dtk | 17.8 dtk |
| Terlama | 97.5 dtk | 26.5 dtk |
| Panggilan LLM per request | 2-10 | 1 |
| Contoh "ayat tentang riba" | 22.7 dtk | 12.8 dtk |
| Contoh "hadits senyum sedekah" | 61.6 dtk | 11.5 dtk |

Dengan semantik aktif, satu contoh ("ayat tentang bersabar menghadapi kesulitan
hidup") total 17.2 dtk dengan retrieval 1.57 dtk, dan keempat rujukan
terverifikasi `ReferenceValidator`.

Kualitas baru diperiksa pada beberapa jawaban (rujukan terverifikasi, model jujur
saat dalil tidak ditemukan). Belum ada evaluasi sistematis.

## 5. Berkas yang berubah atau ditambah

- Baru: `retrieval.py`, `semantic.py`, `semantic_index.py`, `hadith_repair.py`,
  `tests/test_hadith_repair.py`,
  `tests/test_retrieval.py`, `tests/test_semantic.py`, dokumen ini.
- Diubah: `agent.py`, `config.py`, `.env.example`, `requirements.txt` (numpy),
  `tests/test_agent.py`, `tests/test_streaming.py`.
- Tambahan: `eval/` (dataset dan skrip evaluasi retrieval).
- 146 test lolos.

## 6. Pengaturan baru (`.env`)

| Variabel | Default | Fungsi |
|---|---|---|
| `AGENT_PIPELINE_ENABLED` | true | false = selalu agent loop lama |
| `AGENT_PARALLEL_TOOL_CALLS` | true | false jika gateway menolak parameter ini |
| `MCP_HEALTH_CHECK_INTERVAL_SECONDS` | 30 | 0 = cek tiap request |
| `SEMANTIC_SEARCH_ENABLED` | true | false = hanya pencocokan kata |
| `SEMANTIC_DB_PATH` | `.data/semantic.db` | lokasi indeks vektor |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `text-embedding-3-small` / 512 | harus sama dengan saat indeks dibangun |

## 7. Batasan yang diketahui

- Satu panggilan LLM masih 8-25 dtk (teks pertama muncul di detik 6-22). Ini
  sekarang hampir seluruh waktu, ditentukan model dan gateway.
- Jawaban pipeline tidak menyertakan teks Arab.
- Retrieval masih lemah untuk pertanyaan topikal pendek. Pada evaluasi 19 kasus,
  dalil target masuk 12 kandidat hanya 4/19 (typed) dan 6/19 (neutral); jauh
  lebih baik (12/19) bila pertanyaan menggambarkan isi dalil. Sebagian karena
  pertanyaan seperti "ayat tentang sabar" punya puluhan dalil yang sama benarnya,
  jadi satu dalil target adalah ukuran yang ketat. Hadits yang diriwayatkan
  dengan banyak jalur juga bersaing satu sama lain (contoh: Bukhari #1 niat).
- Panggilan embedding masih sesekali timeout (1 dari ~57 pada evaluasi).
- Pesan "terima kasih" setelah percakapan tetap menjalankan retrieval (topik dipinjam dari
  pertanyaan sebelumnya), menambah sekitar 1 dtk tanpa manfaat. Jawabannya tetap benar.
- Ekstraksi rujukan sesekali ikut menangkap kata sebelum nama surah (contoh label
  "Ada juga Al-Baqarah 2:278"); sudah ada sebelumnya, dan rujukannya tetap terverifikasi.
- Sekitar 269 hadits tidak punya teks sama sekali dan tidak bisa dicari.
- Server harus di-restart agar kode baru aktif. Indeks semantik dimuat saat
  request semantik pertama.

## 8. Target perbaikan berikutnya

Urut prioritas.

0. ~~Pertanyaan lanjutan~~ dan ~~kesiapan deploy indeks semantik~~ Selesai (Langkah 7 dan 8).
1. ~~Embed query yang sudah dibersihkan~~ Sudah diuji dan ditolak (lihat Langkah 4).
2. ~~Perbanyak kandidat semantik~~ Selesai: 12 per sumber.
3. ~~Set evaluasi retrieval~~ Selesai (Langkah 5).
4. ~~Gabungkan peringkat kata dan semantik~~ Selesai: selang-seling (RRF seimbang).
5. **Reranking.** Hanya jika dalil benar sering sudah masuk kandidat tetapi
   diabaikan model. Evaluasi menunjukkan masalah utama ada sebelum itu: dalil
   target sering tidak masuk kandidat sama sekali, jadi prioritaskan poin 5a-5c. Reranking dengan LLM terpisah
   menambah 5-15 dtk dan tidak disarankan. Gateway tidak menyediakan model rerank.
5a. **(Prioritas rendah, hanya jika ada keluhan dalil tidak relevan.)** Perbaiki
    kualitas retrieval pertanyaan pendek. Opsi: perluas query
    dengan sinonim atau deskripsi isi (gaya `gist` jauh lebih akurat), tambah
    sumber embedding (teks Arab, atau terjemahan Inggris untuk ayat), model
    embedding lebih besar (`text-embedding-3-large` tersedia di gateway). Ukur
    ulang dengan `eval/`.
5b. **Perluas set evaluasi.** 19 kasus terlalu sedikit untuk membedakan perubahan
    kecil; tambah dalil alternatif yang sama benarnya per topik, dan kasus dengan
    jawaban "tidak ditemukan".
5c. **Retry embedding sekali** untuk timeout sesekali, dengan batas waktu total.
6. **Kecepatan LLM.** Coba model lain, turunkan reasoning effort, batasi panjang
   jawaban, atau gunakan gateway yang lebih cepat. Bandingkan dengan baris
   `timings`.
7. **Teks Arab.** Tempelkan teks Arab ayat lewat kode setelah jawaban selesai,
   tanpa melalui LLM.
8. ~~Perbaiki database hadits~~ Selesai (Langkah 6). Sisa: perbaiki seeder Go di
   repo `hadith-api-go-run` agar tidak bergantung pada perbaikan pasca-seed.
9. **Cache.** Normalisasi lebih agresif (buang stopword dan kata sopan), dan izinkan
   cache saat ada riwayat jika pertanyaannya berdiri sendiri.
10. **Dev.** Gunakan binary hasil `go build` alih-alih `go run ./cmd/mcp` di
    `runtime_config.json` lokal untuk mempercepat spawn dan reconnect MCP.
