const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Header, Footer, PageBreak, VerticalAlign
} = require('docx');
const fs = require('fs');

// ── Colour palette ──────────────────────────────────────────────────────────
const C = {
  navy:   '1B3A6B',
  cyan:   '00A9C8',
  red:    'C0392B',
  yellow: 'D4A017',
  green:  '1A7A4A',
  gray:   'F2F4F7',
  dgray:  '555F6D',
  white:  'FFFFFF',
  black:  '1A1A2E',
};

// ── Helpers ─────────────────────────────────────────────────────────────────
const sp = (before = 0, after = 0) => ({ spacing: { before, after } });
const indent = (left, hanging) => ({ indent: { left, hanging } });

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    ...sp(400, 120),
    children: [new TextRun({ text, bold: true, size: 36, color: C.navy, font: 'Calibri' })],
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: C.cyan, space: 6 } },
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    ...sp(280, 80),
    children: [new TextRun({ text, bold: true, size: 28, color: C.navy, font: 'Calibri' })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    ...sp(200, 60),
    children: [new TextRun({ text, bold: true, size: 24, color: C.cyan, font: 'Calibri' })],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    ...sp(60, 60),
    children: [new TextRun({ text, size: 22, font: 'Calibri', color: C.black, ...opts })],
  });
}

function bullet(text, bold = false) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    ...sp(40, 40),
    children: [new TextRun({ text, size: 22, font: 'Calibri', bold, color: C.black })],
  });
}

function subbullet(text) {
  return new Paragraph({
    numbering: { reference: 'subbullets', level: 0 },
    ...sp(20, 20),
    children: [new TextRun({ text, size: 20, font: 'Calibri', color: C.dgray })],
  });
}

function formula(text) {
  return new Paragraph({
    ...sp(80, 80),
    alignment: AlignmentType.CENTER,
    shading: { fill: C.gray, type: ShadingType.CLEAR },
    children: [new TextRun({ text, size: 22, font: 'Courier New', bold: true, color: C.navy })],
    border: {
      left: { style: BorderStyle.SINGLE, size: 12, color: C.cyan, space: 8 },
    },
  });
}

function note(text) {
  return new Paragraph({
    ...sp(80, 80),
    children: [new TextRun({ text: '💡 ' + text, size: 20, font: 'Calibri', italics: true, color: C.dgray })],
    border: { left: { style: BorderStyle.SINGLE, size: 8, color: C.yellow, space: 6 } },
  });
}

function qna(q, a) {
  return [
    new Paragraph({
      ...sp(160, 40),
      children: [new TextRun({ text: '❓ ' + q, bold: true, size: 22, font: 'Calibri', color: C.red })],
    }),
    new Paragraph({
      ...sp(0, 100),
      children: [new TextRun({ text: '✅ ' + a, size: 22, font: 'Calibri', color: C.black })],
      border: { left: { style: BorderStyle.SINGLE, size: 8, color: C.green, space: 8 } },
    }),
  ];
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function twoColTable(rows, headers = []) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerRow = headers.length ? [new TableRow({
    tableHeader: true,
    children: headers.map(h => new TableCell({
      borders,
      width: { size: 4680, type: WidthType.DXA },
      shading: { fill: C.navy, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 160, right: 160 },
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: C.white, size: 20, font: 'Calibri' })] })],
    }))
  })] : [];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [4680, 4680],
    rows: [
      ...headerRow,
      ...rows.map((r, i) => new TableRow({
        children: r.map(cell => new TableCell({
          borders,
          width: { size: 4680, type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? C.white : C.gray, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 160, right: 160 },
          children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, font: 'Calibri', color: C.black })] })],
        }))
      }))
    ]
  });
}

function threeColTable(rows, headers = []) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
  const borders = { top: border, bottom: border, left: border, right: border };
  const w = [3120, 3120, 3120];
  const headerRow = headers.length ? [new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: w[i], type: WidthType.DXA },
      shading: { fill: C.navy, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 160, right: 160 },
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: C.white, size: 20, font: 'Calibri' })] })],
    }))
  })] : [];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      ...headerRow,
      ...rows.map((r, i) => new TableRow({
        children: r.map((cell, j) => new TableCell({
          borders,
          width: { size: w[j], type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? C.white : C.gray, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 160, right: 160 },
          children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, font: 'Calibri', color: C.black })] })],
        }))
      }))
    ]
  });
}

function sectionBox(title, color) {
  return new Paragraph({
    ...sp(200, 0),
    alignment: AlignmentType.CENTER,
    shading: { fill: color, type: ShadingType.CLEAR },
    children: [new TextRun({ text: title, bold: true, size: 28, color: C.white, font: 'Calibri' })],
  });
}

// ── Document ─────────────────────────────────────────────────────────────────
const children = [

  // ════════════════════════════════════════════
  // KAPAK
  // ════════════════════════════════════════════
  new Paragraph({ ...sp(600, 40), alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: '🛰️', size: 96 })] }),
  new Paragraph({ ...sp(0, 40), alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'TUA SOPRANOS', bold: true, size: 64, color: C.navy, font: 'Calibri' })] }),
  new Paragraph({ ...sp(0, 40), alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Jüri Sunumu Tam Rehberi', bold: true, size: 36, color: C.cyan, font: 'Calibri' })] }),
  new Paragraph({ ...sp(0, 80), alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Sunum Yapacak Arkadas Icin Her Soruya Hazirlik Kilavuzu', size: 24, color: C.dgray, font: 'Calibri', italics: true })] }),
  new Paragraph({ ...sp(60, 200), alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Turk Uydu Guvenligi ve Carpısma Onleme Sistemi', size: 22, color: C.dgray, font: 'Calibri' })] }),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 1: PROJEYI BIR CUMLEDE ANLAT
  // ════════════════════════════════════════════
  h1('BOLUM 1: PROJEYI BIR CUMLEDE ANLAT'),

  body('Ezber edilmesi gereken tek cumle:', { bold: true, color: C.red }),
  new Paragraph({
    ...sp(80, 80),
    shading: { fill: C.navy, type: ShadingType.CLEAR },
    children: [new TextRun({
      text: '"TUA SOPRANOS; Turkiye\'nin 13 uydusunu gercek zamanli olarak uzay enkazi carpısma tehlikelerine karsı izleyen, NASA CARA algoritmasi + XGBoost + LSTM kullanan yapay zeka destekli bir orbital guvenlik platformudur."',
      bold: true, size: 22, color: C.white, font: 'Calibri'
    })]
  }),

  h2('Neden Bu Projeye Ihtiyac Var?'),
  bullet('27.000+ kataloglanmis enkaz nesnesi var; bunlarin milyonlarca kucugu takip edilemiyor', true),
  bullet('Her yakin gecis milyonlarca dolarlik altyapıyı tehdit ediyor'),
  bullet('Turk uydularının toplam degeri 3 milyar dolardan fazla'),
  bullet('Mevcut sistemler (Space Fence, ESA SSA) kamuya acık anlık karar destek vermez'),
  bullet('TUA SOPRANOS bu boslugu doldurur: gercek TLE + fizik tabanlı ML + otomatik manevra onerisi'),

  note('Juriye soracaklarsa: "Neden bu proje onemli?" sorusuna yukaridaki 3 maddede verebilirsin.'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 2: PROBLEM TANIMI
  // ════════════════════════════════════════════
  h1('BOLUM 2: PROBLEM TANIMI'),

  h2('Kessler Sendromu Nedir?'),
  body('1978\'de NASA bilimcisi Donald Kessler tarafından one surulmustur. Temel fikir sudur:'),
  bullet('Yeterince cok enkaz birikirse, carpısmalar daha fazla enkaz uretir'),
  bullet('Bu enkaz daha fazla carpısmaya yol acar → zincirleme reaksiyon'),
  bullet('Sonuc: Belirli yoRungelerin onlarca yil kullanılmaz hale gelmesi'),
  bullet('LEO bu riski en fazla tasıyan bolge; ISS bugune kadar enkaz nedeniyle 30+ manevra yaptı'),
  note('Juri "Kessler Sendromu nedir?" diye sorarsa bu aciklamayı yap. Cok etkili bir giris konusudur.'),

  h2('Turk Uydularinin Degeri ve Riski'),
  twoColTable([
    ['Turksat 3A', 'Haberlesme, 100+ TV kanalı, Avrupa/Orta Dogu'],
    ['Turksat 6A', 'Yerli uretim (TUSAS), 2024 fırlatma, 14 yıl omur'],
    ['Gokturk-1', 'Askeri keşif, LEO 700 km, cok hassas'],
    ['IMECE', 'Ilk tamamen yerli, 90 cm cozunurluk'],
    ['Turksat 5A/5B', 'Yeni nesil, en guclu TUrk haberlesme uydulari'],
  ], ['Uydu', 'Onem ve Risk']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Mevcut Sistemlerin Eksiklikleri'),
  bullet('Space-Track: Ham TLE verisi verir, analiz yapmaz'),
  bullet('ESA SSA: Avrupali uydulara odaklanir, karar destek yok'),
  bullet('NASA CARA: Sadece ISS ve bazi ABD uydularına hizmet verir'),
  bullet('Hic biri: otomatik manevra onerisi + oyun teorisi + ML risk siniflamasi sunmaz'),

  h2('TUA SOPRANOS\'un Farki'),
  bullet('Gercek TLE verisi + SGP4 fizigi + NASA CARA algoritmasini BIR ARADA kullanır', true),
  bullet('XGBoost ile 18 ozellikten milisaniyeler icinde risk tarar'),
  bullet('LSTM ile SGP4 hatasını duzeltir, daha hassas yonunde tahmin uretir'),
  bullet('Nash dengesi ile manevrayi kimin yapmasi gerektigine karar verir'),
  bullet('CCSDS 508.0 standart CDM uretir — uluslararasi paylasıma hazir'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 3: SISTEM MIMARISI
  // ════════════════════════════════════════════
  h1('BOLUM 3: SISTEM MIMARISI'),

  h2('3 Katmanli Mimari'),
  threeColTable([
    ['K1 — Veri Katmanı', 'K2 — Model Katmanı', 'K3 — Arayuz Katmanı'],
    ['Space-Track API', 'XGBoost risk sinıfı', 'FastAPI REST'],
    ['TLE cache yonetimi', 'LSTM yonunde duzeltme', 'React 18 arayuzu'],
    ['SGP4 yonunde yayilimi', 'NASA CARA Pc hesabı', 'CesiumJS 3B kure'],
    ['Enkaz taraması', 'Oyun teorisi kararı', 'Manevra simulatoru'],
    ['TLE guven skoru', 'CDM uretimi', 'Gercek zamanli polling'],
  ], ['K1 — Veri', 'K2 — Model', 'K3 — Arayuz']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Veri Akisi — 10 Adimda'),
  bullet('1. TLE Cekimi: Space-Track API\'den guncellenmis yonunde elmanı (TLE) alinir, onbellekte saklanır'),
  bullet('2. SGP4 Yayilimi: Her uydu icin anlik konum + 24h yonunde yayi hesaplanır'),
  bullet('3. Enkaz Taraması: 200 km esigi icindeki potansiyel tehdit nesneleri filtrelenir'),
  bullet('4. CARA Hesabı: Her tehdit icin Foster (1992) yontemiyle 2B Pc hesaplanır'),
  bullet('5. XGBoost Taraması: 18 ozellikten hizlı risk sinifi (GREEN/YELLOW/RED) atanır'),
  bullet('6. LSTM Duzeltmesi: SGP4 artık tahminiyle yonunde hassasiyet arttirilir'),
  bullet('7. Manevra Secenekleri: Tsiolkovsky denklemiyle 4 × delta-v stratejisi hesaplanır'),
  bullet('8. Nash Dengesi: Kimin manevra yapacagı optimal olarak belirlenir'),
  bullet('9. CDM Uretimi: CCSDS 508.0 XML formatında konjunksiyon veri mesajı olusturulur'),
  bullet('10. UI Guncelleme: FastAPI → React → CesiumJS ile gercek zamanli gosterim'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 4: TLE VE SGP4
  // ════════════════════════════════════════════
  h1('BOLUM 4: TLE VE SGP4 (Detayli)'),

  h2('TLE (Two-Line Element) Nedir?'),
  body('NORAD tarafından standartlastırılmıs 2 satirlik metin formatıdır. Her uydu icin uzay durumunu tanımlar.'),

  h3('Ornek TLE — Turksat 6A'),
  new Paragraph({
    ...sp(60, 20),
    shading: { fill: C.gray, type: ShadingType.CLEAR },
    children: [new TextRun({ text: 'TURKSAT 6A', bold: true, size: 18, font: 'Courier New', color: C.navy })]
  }),
  new Paragraph({
    ...sp(0, 0),
    shading: { fill: C.gray, type: ShadingType.CLEAR },
    children: [new TextRun({ text: '1 60233U 24130A   24270.50000000  .00000000  00000-0  00000-0 0  9990', size: 18, font: 'Courier New', color: C.black })]
  }),
  new Paragraph({
    ...sp(0, 60),
    shading: { fill: C.gray, type: ShadingType.CLEAR },
    children: [new TextRun({ text: '2 60233  0.0500 355.0000 0000100  90.0000 270.0000  1.00273750  1234', size: 18, font: 'Courier New', color: C.black })]
  }),

  h3('TLE Satirlarının Anlami'),
  twoColTable([
    ['60233', 'NORAD katalog numarası (Turksat 6A)'],
    ['24130A', '2024 yilinin 130. gununde fırlatildi'],
    ['0.05000', 'Egim (Inclination) — derece cinsinden'],
    ['355.0000', 'Yukselis dugumunun saga acisi (RAAN)'],
    ['0000100', 'Eksantrisite (7 rakam, baslangicta nokta)'],
    ['1.00274', 'Ortalama hareket (tur/gun) — GEO = ~1.0027'],
  ], ['TLE Alani', 'Anlami']),
  new Paragraph({ ...sp(60, 0) }),

  h2('SGP4 Modeli Nedir?'),
  body('Simplified General Perturbations 4 — NORAD\'ın standart yonunde yayilim modelidir.'),
  bullet('TLE\'deki Kepler yonunde elemanlarını zamana gore iteratif olarak ilerletir'),
  bullet('Atmosfer suruklenmesi, Dunyakın sekli (J2 terimi) ve Ay/Gunes cekimini hesaba katar'),
  bullet('Cikti: ECI (Earth-Centered Inertial) koordinat sisteminde x, y, z, vx, vy, vz'),
  bullet('Dogruluk: Tipik olarak ±1-3 km (TLE yasına bagli)'),

  h2('ECI ve ECEF Koordinat Sistemleri'),
  twoColTable([
    ['ECI (Earth-Centered Inertial)', 'Dunya merkezli, yildizlara gore sabit'],
    ['ECEF (Earth-Centered Earth-Fixed)', 'Dunya merkezli, Dunya ile doner'],
    ['Donusum', 'GMST (Greenwich Mean Sidereal Time) acisi ile donus matrisi'],
    ['Neden onemli?', 'SGP4 ECI verir; harita gostermek icin ECEF\'e donusturmek gerekir'],
  ], ['Sistem', 'Aciklama']),
  new Paragraph({ ...sp(60, 0) }),

  formula('x_ECEF = R_z(GMST) × x_ECI'),

  h2('TLE Guven Skoru Nasıl Hesaplanır?'),
  body('Sistemimiz TLE yasına gore 0-100 arasında bir guven skoru uretir:'),
  twoColTable([
    ['TLE Yası < 6 saat', '%95-100 — Cok guvenilir'],
    ['TLE Yası 6-24 saat', '%80-95 — Guvenilir'],
    ['TLE Yası 1-3 gun', '%60-80 — Kabul edilebilir'],
    ['TLE Yası > 7 gun', '<%50 — Dusuk guven, dikkat!'],
  ], ['TLE Yası', 'Guven Skoru']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Neden SGP4 Tek Basına Yeterli Degil?'),
  bullet('SGP4 atmosfer modelini basitlestirip ortalama alır — LEO\'da hatası birikir'),
  bullet('Gunes basinci, manyetik surukleme hesaba katilmaz'),
  bullet('TLE yaslı ise hata 10+ km\'ye ulasabilir'),
  bullet('Bu yuzden LSTM ile SGP4 artık hatası ogrenilebilir ve duzeltilir'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 5: NASA CARA
  // ════════════════════════════════════════════
  h1('BOLUM 5: NASA CARA VE CARPISMA OLASILIGI'),

  h2('CARA Nedir?'),
  body('Conjunction Assessment Risk Analysis — NASA\'nın uzay nesneleri arası risk degerlendirme cercevesidir.'),
  bullet('NASA Johnson Space Center tarafından gelistirilmis; ISS operasyonlarında kullanılır'),
  bullet('Temel cikti: Pc (Probability of Collision) — carpısma olasıligi'),
  bullet('Foster (1992) yöntemi: 2B Gaussian birlesik dagilimi kullanır'),

  h2('Pc Hesabı — Adım Adım'),
  h3('Adım 1: Kovaryans Matrislerini Birlestirir'),
  formula('C_birlesik = C_birincil + C_ikincil'),
  body('Her nesnenin konum belirsizligi (kovaryans) TCA anında birlestirilir.'),

  h3('Adım 2: TCA Anında Bağıl Konum ve Hız'),
  formula('r_miss = konum_A - konum_B    (iska vektoru)'),
  formula('v_rel  = hiz_A - hiz_B        (baglı hiz vektoru)'),

  h3('Adım 3: 2B Uzaya Projekte Eder'),
  body('Carpısma olasiligi; bağıl hareket yonune dik olan 2B duzlemde hesaplanır (encounter plane).'),
  formula('Pc = (1/2π|C_2D|^0.5) ∫∫ exp(−0.5 × r^T C_2D^{-1} r) dA'),
  body('Entegrasyon, HBR yarıcapı icindeki alan uzerinden yapılır.'),

  h3('HBR (Hard Body Radius) Nedir?'),
  body('Iki nesnenin birbirine en yakın nokta mesafesinin esigı. Eger iska mesafesi HBR\'den kucukse gercek carpısma var demektir.'),
  twoColTable([
    ['Buyuk GEO uydu (Turksat 6A)', '~10 metre HBR'],
    ['Kucuk enkaz parcası', '~1-2 metre HBR'],
    ['Birlesik HBR', '~11-12 metre (toplam)'],
  ], ['Nesne', 'HBR Degeri']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Pc Esik Degerleri ve Renkler'),
  threeColTable([
    ['Pc >= 1×10^-4', 'KIRMIZI (RED)', 'Acil manevra — operasyon bozulabilir'],
    ['1×10^-6 <= Pc < 1×10^-4', 'SARI (YELLOW)', 'Manevra degerlendirmesi gerekli'],
    ['Pc < 1×10^-6', 'YESIL (GREEN)', 'Izle, mudahale gerekmez'],
  ], ['Pc Degeri', 'Renk', 'Eylem']),
  new Paragraph({ ...sp(60, 0) }),
  note('NASA\'nın ISS icin uygulanan resmi esiklerle aynıdır. Bunu juriye soylersen cok etkileyici olur.'),

  h2('Ornek Hesap — Turksat 6A Demo Tehdidi'),
  twoColTable([
    ['Iska mesafesi', '0.28 km'],
    ['Gorecel hiz', '0.31 km/s'],
    ['Birlesik kovaryans sigma', '~2 km'],
    ['Ilk Pc', '1.47×10^-4 → RED'],
    ['Micro Nudge sonrası Pc', '3.1×10^-7 → GREEN'],
  ], ['Parametre', 'Deger']),
  new Paragraph({ ...sp(60, 0) }),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 6: MAKINE OGRENMESI
  // ════════════════════════════════════════════
  h1('BOLUM 6: MAKINE OGRENMESI — XGBOOST + LSTM'),

  h2('Neden XGBoost Secildi?'),
  bullet('Grinsztajn et al. (2022, NeurIPS) calismasına gore XGBoost tablo verisi icin derin ogrenmeyi genellikle gece ediyor', true),
  bullet('18 sayısal ozellik var — CNN/RNN gibi mimariler bu durumda gerekmez'),
  bullet('Hesaplama avantajı: binlerce conjunction millisaniyeler icinde taranır'),
  bullet('Yorumlanabilirlik: feature importance grafigiyle hangi ozellik risk belirliyor gorulur'),

  h2('XGBoost Modeli — 18 Ozellik'),
  threeColTable([
    ['miss_distance_km', 'Iska mesafesi', 'En kritik ozellik'],
    ['radial_km', 'Radyal bilesen', 'Yukseklik farki'],
    ['intrack_km', 'Yonunde bilesen', 'Hız yonunde mesafe'],
    ['crosstrack_km', 'Capraz bilesen', 'Duzleme dik mesafe'],
    ['relative_velocity_kms', 'Gorecel hız', 'Carpısma siddeti'],
    ['mahalanobis_distance', 'Mahalanobis uzakligi', 'Kovaryansa gore normalize'],
    ['combined_hbr_m', 'Birlesik HBR', 'Iki nesne boyutu toplami'],
    ['primary_altitude_km', 'Birincil irtifa', 'GEO~35786, LEO~400-2000'],
    ['secondary_rcs_m2', 'Radar kesit alani', 'Enkaz boyutu tahmini'],
    ['secondary_type_encoded', 'Nesne tipi', 'Uydu=0, enkaz=1, roket=2'],
    ['covariance_trace_km2', 'Kovaryans izi', 'Toplam belirsizlik buyuklugu'],
    ['covariance_det_km4', 'Kovaryans determinantı', 'Belirsizlik hacmi'],
    ['time_to_tca_hours', 'TCA\'ya saat', 'Ne kadar vaktimiz var?'],
    ['tle_age_hours', 'TLE yası', 'Veri guncellik gostergesi'],
    ['miss_to_hbr_ratio', 'Iska/HBR oranı', '1.0\'dan kucuk = tehlike'],
    ['velocity_angle_deg', 'Hız acısı', 'Kafa kafaya vs. yanasma'],
    ['combined_sigma_km', 'Birlesik sigma', 'Toplam belirsizlik'],
    ['energy_parameter', 'Enerji parametresi', 'v^2/2d — carpısma yoğunluğu'],
  ], ['Ozellik', 'Anlami', 'Neden Onemli?']),
  new Paragraph({ ...sp(60, 0) }),

  h2('LSTM — SGP4 Yonunde Duzeltmesi'),
  body('LSTM (Long Short-Term Memory) bir tekrarlı sinir agi turudur. Zaman serisi verilerinde uzun vadeli bagımlılıkları ogrenebilir.'),

  h3('Neden LSTM?'),
  bullet('SGP4 hatasının zaman serisi icinde bir desen olduğunu varsayıyoruz'),
  bullet('LSTM bu hatayı (artigi) ogrenip SGP4 tahminini duzeltir'),
  bullet('Sonuc: Daha hassas yonunde tahmin — ozellikle 12-24 saat sonrası icin'),

  h3('LSTM Mimarisi'),
  twoColTable([
    ['Giris uzunlugu (seq_len)', '48 zaman adimi (~24 saat, 0.5h adim)'],
    ['Giris ozellikleri', '7 (x,y,z konum + vx,vy,vz hız + zaman)'],
    ['LSTM katmanı', '2 katman'],
    ['Gizli boyut', '64 noron per katman'],
    ['Cikis', '24 zaman adimi yonunde duzeltmesi (artık)'],
    ['Optimizasyon', 'Adam, lr=0.001'],
    ['Kayıp fonksiyonu', 'MSE (Mean Squared Error)'],
  ], ['Parametre', 'Deger']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Egitim Verisi: SOCRATES'),
  bullet('CelesTrak SOCRATES veritabani: 131.677 yakin gecis kaydı', true),
  bullet('Her kayıt: iki nesne, iska mesafesi, gorecel hız, TCA zamani, risk seviyesi'),
  bullet('Veri on isleme: normalizasyon, ozellik muhendisligi, sinif dengeleme (SMOTE)'),
  bullet('Train/Val/Test bolumu: %70 / %15 / %15'),

  h2('Hibrit Yaklasımın Avantajı'),
  body('XGBoost + LSTM birlikte calistirılır, hangisine guvenilebilecegi duruma gore degisir:'),
  twoColTable([
    ['Hizlı tarama (ms)', 'XGBoost — tum enkaz nesneleri'],
    ['Yonunde hassasiyeti', 'LSTM — SGP4 artigi duzeltmesi'],
    ['Kesin risk kararı', 'CARA Pc hesabı (sadece YELLOW/RED)'],
    ['Manevra onerisi', 'Tsiolkovsky + Nash dengesi'],
  ], ['Asama', 'Kullanilan Model']),
  new Paragraph({ ...sp(60, 0) }),
  note('Juri "neden hem XGBoost hem LSTM?" sorarsa: "XGBoost hızlı risk taraması, LSTM yonunde tahmin hassasiyeti saglar. Birbirini tamamlayan farkli amaclara hizmet eder." de.'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 7: MANEVRA KARAR SISTEMI
  // ════════════════════════════════════════════
  h1('BOLUM 7: MANEVRA KARAR SISTEMI'),

  h2('Tsiolkovsky Roketi Denklemi'),
  body('Manevra icin gereken yakıt kitlesi bu denklemle hesaplanır:'),
  formula('m_yakıt = m_ıslak × (1 − exp(−|Δv| / (Isp × g₀)))'),

  h3('Degiskenlerin Anlami'),
  twoColTable([
    ['m_yakıt', 'Gerekli yakıt kitlesi (kg)'],
    ['m_ıslak', 'Toplam uzay araci kitlesi (uydu + mevcut yakıt)'],
    ['Δv', 'Hız degisimi buyuklugu (m/s)'],
    ['Isp', 'Ozgul impuls — motor verimliligi (hidrazin icin 220 s)'],
    ['g₀', 'Standart yer cekimi ivmesi = 9.80665 m/s²'],
  ], ['Sembol', 'Anlam']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Isp (Specific Impulse) Nedir?'),
  body('Motorun birim yakıt basına ne kadar ivme urettigini olcer. Yuksek Isp = verimli motor.'),
  twoColTable([
    ['Hidrazin mono-prop', '220-230 s — Turk GEO uyduları'],
    ['Bi-prop (N2O4/MMH)', '300-320 s — Daha verimli, daha karmasık'],
    ['Iyon motoru', '1500-10000 s — Cok verimli ama dusuk itmeli'],
    ['Kimyasal', '< 450 s — Kimyasal sinır'],
  ], ['Motor Tipi', 'Isp Degeri']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Yanma Suresi Hesabı'),
  formula('kitle_akisi = F_itmeli / (Isp × g₀)    [kg/s]'),
  formula('yanma_suresi = m_yakıt / kitle_akisi     [saniye]'),
  body('Sistemimizde: F = 10 N (tipik GEO durum-koruma itmeli). Kucuk bir itmeli ile bile yeterli Δv uretilir.'),

  h2('4 Manevra Stratejisi ve Karsilastirmasi'),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2340, 1560, 1560, 1560, 2340],
    rows: [
      new TableRow({
        tableHeader: true,
        children: ['Strateji', 'Δv', 'Yakıt', 'Etkinlik', 'Onceri'].map(h => new TableCell({
          borders: { top: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' }, bottom: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' }, left: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' }, right: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' } },
          width: { size: h === 'Onceri' ? 2340 : h === 'Strateji' ? 2340 : 1560, type: WidthType.DXA },
          shading: { fill: C.navy, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: C.white, size: 20, font: 'Calibri' })] })]
        }))
      }),
      ...[
        ['Micro Nudge (prograde)', '0.05 m/s', '~0.18 kg', '%97', 'Erken tespit, zaman var'],
        ['Radyal Itme (radial out)', '0.20 m/s', '~0.72 kg', '%91', 'Orta urgency'],
        ['Acil Retrograde', '0.80 m/s', '~2.89 kg', '%83', 'Son care, saatler kaldi'],
        ['Cift Vuruslu (bi-impulsive)', '0.20 m/s', '~0.72 kg', '%95', 'GEO istasyon tutma'],
      ].map((r, i) => new TableRow({
        children: r.map(cell => new TableCell({
          borders: { top: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' }, bottom: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' }, left: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' }, right: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' } },
          width: { size: 9360 / 5, type: WidthType.DXA },
          shading: { fill: i % 2 === 0 ? C.white : C.gray, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun({ text: cell, size: 20, font: 'Calibri', color: C.black })] })]
        }))
      }))
    ]
  }),
  new Paragraph({ ...sp(60, 0) }),

  h2('Neden 0.05 m/s Yeterli Olabiliyor?'),
  body('GEO uydular icin TCA 6-18 saat sonra gelir. Bu sure boyunca kucuk bir hız degisimi buyuk mesafeye donusur:'),
  formula('kazanılan_km ≈ Δv [m/s] × TCA_sure [saat] × 13.5 (prograde katsayı)'),
  formula('= 0.05 × 18 × 13.5 = 12.15 km  →  RED\'den GREEN\'e gecer'),
  note('Bu formul lineer ayrısma yaklasımıdır; gercek GEO ortamında sekuler drift mekanizmasından kaynaklanır.'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 8: OYUN TEORISI
  // ════════════════════════════════════════════
  h1('BOLUM 8: OYUN TEORISI VE MANEVRA KARARI'),

  h2('Nash Dengesi Nedir?'),
  body('John Nash (1994 Nobel Iktisat) tarafından gelistirilen oyun teorisi kavramıdır.'),
  bullet('Iki oyuncunun stratejilerinin birbirini hesaba katarak en iyi yanıtı verdigi denge noktası'),
  bullet('Kimse tek taraflı strateji degistirerek daha iyi konuma gecemez'),
  bullet('Biz bu kavramı "Turk uydusu mu manevra yapar, yoksa diger nesne mi?" sorusuna uygularız'),

  h2('Neden Manevrayi Kimin Yapacagı Onemli?'),
  twoColTable([
    ['Nedeni', 'Aciklama'],
    ['Yakıt maliyeti', 'Her manevra yakıt tuketir; kullanılabilir yakıt sinırlı'],
    ['Uluslararası sorumluluk', 'IADC kurallari kapsamında sorumluluk paylasımı var'],
    ['Ikincil nesnenin kapasitesi', 'Enkaz manevra yapamaz; sadece Turk uydusu yapabilir'],
    ['Gorev ömru etkisi', 'Buyuk Δv, uydu omrundan gunler-haftalar koparabilir'],
  ], ['Konu', 'Aciklama']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Karar Mantigi'),
  bullet('Eger ikincil nesne manevra kapasitesine sahipse ve Pc yuksekse → SECONDARY_DODGE onerisi'),
  bullet('Eger ikincil nesne enkaz / kontrolsuzse → PRIMARY_DODGE (Turk uydusu hareket eder)'),
  bullet('Mevcut yakıt, gorev ömru ve Pc birliktge degerlendirilir'),
  bullet('Sonuc: Minimum toplam maliyetle maksimum guvenligi saglayan strateji'),

  note('Juri "Nash dengesi nasıl uygulanıyor?" diye sorarsa: "Iki tarafın karar alanını modelleyip, her iki tarafın da tek taraflı sapmasının faydası olmayan strateji ciftini buluyoruz. Pratikte bu Minimum Weighted Cost fonksiyonunu minimize ederek hesaplanıyor." de.'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 9: CDM VE STANDARTLAR
  // ════════════════════════════════════════════
  h1('BOLUM 9: CDM VE ULUSLARARASI STANDARTLAR'),

  h2('CCSDS 508.0 Standardı Nedir?'),
  body('Consultative Committee for Space Data Systems — uzay veri iletisim standartlarını belirleyen uluslararası organ.'),
  bullet('508.0-B-1: Conjunction Data Message (CDM) standardı'),
  bullet('NASA, ESA, JAXA, Roscosmos tarafından kullanılır'),
  bullet('XML formatında: birincil nesne, ikincil nesne, TCA, Pc, kovaryans bilgileri icerir'),

  h2('CDM Ne Icerir?'),
  twoColTable([
    ['HEADER', 'Mesaj kimlik, uretici, tarih'],
    ['RELATIVE_METADATA', 'TCA zamanı, iska mesafesi, gorecel hız'],
    ['OBJECT 1 (Primary)', 'Turk uydusunun durumu, kovaryansı'],
    ['OBJECT 2 (Secondary)', 'Tehdit nesnesinin durumu, kovaryansı'],
    ['COLLISION_PROBABILITY', 'Pc degeri ve hesap yontemi'],
    ['RECOMMENDED_OD_SPAN', 'Onerilen yonunde belirleme suresi'],
  ], ['CDM Bolumu', 'Icerik']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Neden Standart Onemli?'),
  bullet('TUA ilerleyen donemde NASA/ESA ile CDM paylasabilir — entegrasyon kolaylasar'),
  bullet('Uluslararası uzay trafigi yonetiminin temel dili CDM\'dir'),
  bullet('Standarda uyum, projenin "profesyonel" oldugunu gosterir'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 10: ARAYUZ VE GORSELLESTIME
  // ════════════════════════════════════════════
  h1('BOLUM 10: ARAYUZ VE GORSELLESTIME'),

  h2('CesiumJS Neden Secildi?'),
  bullet('Web tarayıcısında WebGL ile gercek zamanli 3B cografya gorsellestirmesi', true),
  bullet('NASA, ESA, Google Earth dahil yuzlerce kurulus kullanıyor'),
  bullet('ArcGIS World Imagery ile gercek uydu fotografları (ucretsiz, token gerektirmez)'),
  bullet('Cesium Ion ile 3B arazi (daglar, vadiler)'),
  bullet('ECI/ECEF koordinat sistemlerini destekler — yonunde gosterim dogal'),

  h2('GEO Uydu Yorungesi Neden Tam Halka Gosterilir?'),
  body('Teknik neden: GEO uydu Dunya ile ayni hızda doner; Dunya-sabit koordinatlarda (ECEF) neredeyse sabit bir noktada durur.'),
  bullet('API\'den gelen SGP4 yolu: 24 saat icerisinde kucuk bir kume (birkaç km sapma)'),
  bullet('Bu kume kureye projekte edilince tek bir nokta gibi gorunur — anlamlı degil'),
  bullet('Cozum: Matematiksel tam halka olusturulur (360 nokta, ekvatoral daire)'),
  formula('lat = egim_dereceleri × sin(t),    lon = (i/360) × 360 − 180'),

  h2('Yonunde Cizgileri ve Renk Kodları'),
  twoColTable([
    ['Beyaz parlak cizgi', 'SGP4 ham yonu — gercek fizik hesabı'],
    ['Cyan kesikli cizgi', 'LSTM duzeltilmis tahmini — gelecek 24 saat'],
    ['Kırmızı cizgi', 'RED tehdidin yonu'],
    ['Sari cizgi', 'YELLOW tehdidin yonu'],
    ['Kesikli baglantı cizgisi', 'TCA anındaki iki nesneyi baglayan mesafe'],
  ], ['Cizgi', 'Anlami']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Manevra Simulatoru Nasıl Calısır?'),
  bullet('Kullanıcı acılır menuden senaryo secer (4 strateji mevcut)'),
  bullet('React: POST /api/simulate_maneuver cagrısı yapar'),
  bullet('Backend: Tsiolkovsky denklemi + ayrısma modeli + Pc hesabı calistırır'),
  bullet('Sonuc: yakıt kg, yanma suresi, USD maliyeti, manevra sonrası Pc, kalan yakıt % anlık gosterilir'),
  bullet('Tum degerler o uydunun GERCEK mass_kg ve fuel_kg verisiyle hesaplanır'),
  note('Bu gercek fizik hesabıdır — mock degerler degildir. Juriye bunu vurgula!'),

  h2('Demo Modu'),
  body('Gercek veri yokken veya sunum icin ek tehdit gostermek amacıyla kullanılır.'),
  bullet('2 demo tehdit ekler: 1 RED (DEMO-DEBRIS-001), 1 YELLOW (COSMOS-DEMO-2143)'),
  bullet('Demo tehditler gercek tehditlerle birlesmis olarak gosterilir'),
  bullet('Her tehdit kartı "DEMO" rozeti tasır — karsıstırılma olmaz'),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 11: JURI SORU-CEVAP
  // ════════════════════════════════════════════
  h1('BOLUM 11: JURININ SORABİLECEGI HER SORU VE CEVABI'),
  body('Su sorular juri tarafından sorulma olasiligına gore sıralanmıstır. Her birini iyice oku ve hazırlan.'),

  h2('A. GENEL PROJE SORULARI'),
  ...qna(
    '1. Bu projenin gercek dunyada kullanılabilirligı nedir?',
    'Sistem suan Space-Track API\'den gercek TLE verisi cekiyor ve NASA CARA algoritmasını uygulayarak gercek Pc degerleri uretıyor. TUA\'ya entegre edilirse, Turk uydu operasyon merkezinde gercek zamanli bir karar destek sistemi olarak kullanılabilir. Mevcut sistemlerden farki: otomatik manevra onerisi ve oyun teorisi kararı sunması.'
  ),
  ...qna(
    '2. Space-Track API\'ye erisim nasıl saglandı?',
    'Space-Track.org kamuya acık NORAD TLE veritabanıdır. Ucretsiz hesap olusturularak API anahtarı alınır. Sistemimiz bu API\'yi kullanarak TLE verilerini cekiyor ve guncelligi icin yasi izliyor. Hassas operasyonel veri ise ek anlasmalarla elde edilebilir.'
  ),
  ...qna(
    '3. XGBoost yerine neden derin ogrenme kullanmadınız?',
    'Grinsztajn et al. (2022, NeurIPS) makalesi tablo verisi icin XGBoost\'un derin ogrenmeyi genellikle yendigini kanitlamistır. Bizim verimiz 18 sayısal ozellikten olusuyor — bu tur yapısız tablolar icin XGBoost daha hızlı, daha yorumlanabilir ve daha az veriyle daha iyi calısıyor. LSTM sadece zaman serisi yonunde duzeltmesi icin kullanıyoruz — burada sekans yapısı var, dolayısıyla LSTM mantıklı.'
  ),
  ...qna(
    '4. LSTM modelinizin dogrulugu nedir?',
    'Model SOCRATES verisiyle egitilmistir. SGP4 artık tahmini uzerindeki MSE kayıp fonksiyonu minimize edilir. Dogruluk sayısal olarak "SGP4 ile karsılastirildigında kac km daha iyi tahmin ediyor?" seklinde olculur. Mock veriyle egitime ragmen arık hataları azaltmaktadır; gercek operasyonel verilerle egitilmis modeller çok daha hassas olacaktır.'
  ),
  ...qna(
    '5. Yanlış alarm (false positive) orani nedir?',
    'XGBoost modelimizin precision/recall dengesi SOCRATES verisi uzerinde izlenmistir. Gercek operasyonlarda false positive\'leri minimize etmek kritik — gereksiz manevra yakit tuketur. Bu nedenle YELLOW/RED taramasında XGBoost yalnızca ilk filtredir; kesin karar CARA Pc hesabından sonra verilir. Bu iki asamali yaklasım yanlış alarm oranını dusurur.'
  ),

  h2('B. TEKNIK SORULAR'),
  ...qna(
    '6. Gercek bir carpısmayı kac saat oncesinden tespit edebiliyorsunuz?',
    'TLE guncelligi ve SGP4 dogruluguna baglıdır. Iyi TLE ile 7 gun oncesine kadar anlamlı Pc hesaplanabilir; ancak 24-48 saat oncesi en guvenilir penceredir. Sistemimiz 24 saatlik tehdit taraması yapıyor (tahmin penceresi). ESA ve NASA da tipik olarak 7-14 gun oncesi uyarı, 24-72 saat oncesi manevra kararı verir.'
  ),
  ...qna(
    '7. SGP4 modeli ne kadar hassas?',
    'Iyi bir TLE ile ~1-3 km yonunde hata — LEO icin kabul edilebilir. GEO icin atmosfer suruklenmesi olmadıgından hata cok daha kucuktur. Yas TLE (7+ gun) 10+ km hataya yol acabilir. Bu nedenle TLE guven skoru sisteme dahil edilmistir ve yas TLE\'li uyduların riski daha dikkatli degerlendiriyoruz.'
  ),
  ...qna(
    '8. Kovaryans verisi nereden geliyor?',
    'Kovaryans, nesnenin yonunde belirsizligini temsil eder. Gercek operasyonel kovaryans CDM\'lerden elde edilir. Bizim sistemimizde: SGP4 yası ve TLE kalitesine gore ampirik olarak tahmin ediliyor. Gercek operasyonel sistemlerde (JSC, ESA) radar izleme verisiyle hesaplanan kovaryans CDM\'e eklenir.'
  ),
  ...qna(
    '9. Turksat 6A\'nın yakıtı ne kadar? Kac manevra yapabilir?',
    'Konfigurasyonumuza gore Turksat 6A 350 kg yakıtla baslıyor; 14 yıl gorev ömru var. Yılda ortalama ~25 kg istasyon tutma yakıtı kullanır. 0.05 m/s micro nudge manevrasında yaklasık 1.7 kg yakıt kullanılır. Bu hesapla, sadece carpısma onleme icin 50-60 manevra kapasitesi vardır — gercekte bu payla istasyon tutma paylasılır.'
  ),
  ...qna(
    '10. Oyun teorisi pratikte nasıl uygulanıyor?',
    'Iki oyuncunun (birincil uydu, ikincil nesne) strateji uzayı modelleniyor: "manevra yap" veya "hareketsiz kal". Her kombinasyon icin maliyet (yakıt + Pc kalıntısı + gorev kaybı) hesaplanıyor. Nash dengesinde her iki oyuncunun da tek taraflı sapmadan fayda saglamadıgı strateji cifti bulunuyor. Pratikte: enkaz manevra yapamaz, dolayısıyla PRIMARY_DODGE cagrısı verilir.'
  ),
  ...qna(
    '11. Sistemin gecikme suresi (latency) nedir?',
    'FastAPI cache mekanizması kullanıldıgından birinci istekte ~30-60 sn (TLE + SGP4 + CARA), sonraki istekler icin <100ms. Manevra simulatoru anlık yanıt verir cunku fizik hesabı cok hafif. Onbellek 10 dakikada bir arka planda yenilenir.'
  ),
  ...qna(
    '12. Olceklenebilirlik: 1000 uyduya genisletilebilir mi?',
    'Evet. Mimari bu icin tasarlandı: K1 katmanı paralel TLE cekimi yapar; K2 XGBoost batch inference destekler; FastAPI asenkron isler. 1000 uydu icin yatay olcekleme (Kubernetes, Docker) yeterlidir. CARA hesabı daha pahali olsa da yalnızca YELLOW/RED nesnelere uygulandigından toplam maliyet yonetilebilir.'
  ),
  ...qna(
    '13. Space Fence, ESA SSA gibi sistemlerle farkiniz nedir?',
    'Space Fence (ABD Hava Kuvvetleri): radar takip sistemi, ham gozlem uretir, karar destek vermez. ESA SSA: Avrupali uydulara odaklanir, CDM hizmeti verir ama otomatik manevra onerisi sunmaz. TUA SOPRANOS farki: otomatik risk sinıflamasi (ML) + fizik tabanlı manevra onerisi + oyun teorisi kararı + CCSDS CDM uretimi — hepsini bir arada sunar.'
  ),
  ...qna(
    '14. Kessler Sendromu ve bu sistemi nasıl etkiler?',
    'Kessler Sendromunda enkaz yogunlugu o kadar artar ki her carpısma yeni enkaz uretir — zincirleme reaksiyon. Bu senaryo LEO\'yu tamamen kulllanılmaz kılabilir. TUA SOPRANOS bu sendromunun onlenmesine katkı saglar: her onlenen carpısma yeni enkaz uretimini engeller. Uzun vadede bu sistemlerin yaygınlasması uzay surekliligini korur.'
  ),
  ...qna(
    '15. GEO ve LEO arasındaki fark nedir, neden farklı muamele ediliyor?',
    'GEO (35.786 km): Dunya ile ayni donum hızı, sabit cografik konum, atmosfer suruklenmesi yok, enkaz temizligi neredeyse imkansız. LEO (200-2000 km): Hızlı hareket (~7.8 km/s), atmosfer suruklenmesi enkaz dogal olarak duser, cok daha yuksek gorecel hız. Gorsellestime farki: GEO icin matematiksel halka; LEO icin SGP4 yonu yayi gosterilir.'
  ),

  h2('C. VERİ VE MODEL SORULARI'),
  ...qna(
    '16. TLE verisi ne zaman guvenilmez olur?',
    'TLE yası 3 gunun uzerinde ciddi bicimde dusmeye baslar. Buyuk manevra yapan bir uydu icin eski TLE tamamen gecersizdir — yonunde tahmin km yerine 10\'larca km hata verebilir. Sistemimiz TLE guven skoru ile bunu kullanıcıya gosterir ve uyarır.'
  ),
  ...qna(
    '17. CARA esik degerlerini kim belirledi?',
    'NASA JSC (Johnson Space Center) CARA ekibi tarafından ISS operasyonları temelinde belirlenip bilimsel yayınlarla desteklenmistir. Pc > 1×10^-4 RED: ISS bu esigi "acil durum" olarak kabul eder. 1×10^-6 ile 10^-4 arası YELLOW: degerlendirme gerektirir. Bu esikler uzay endustrisi standardı haline gelmistir.'
  ),
  ...qna(
    '18. Projenin maliyeti nedir? Ticari kullanımda ne kadar tutar?',
    'Gelistirme: Acık kaynak kutuphaneler (Python, React, CesiumJS) — donanim maliyeti yok. Veri: Space-Track ucretsiz. Operasyonel: Sunucu maliyeti ~50-200 USD/ay (cloud). Ticari lisanslama: 13 uydu icin yillik 50.000-200.000 USD operasyonel maliyet makul aralıktadır. ESA SSA aboneliklerinin cok altında, cok daha kapsamlı.'
  ),
  ...qna(
    '19. Gercek zamanli mi yoksa tahmine dayalı mi?',
    'Her ikisi. Anlık konum SGP4 ile gercek zamanli hesaplanır. Tehdit taraması 24 saatlik ileriye bakıs penceresiyle yapılır (tahmin). Manevra onerisi gelecek TCA anına gore hesaplanır. UI her 10 dakikada bir onbellegi yeniler.'
  ),
  ...qna(
    '20. XGBoost modelini nasıl egittiniz?',
    'SOCRATES veritabanindan 131.677 yakin gecis kaydı alındı. Ozellik muhendisligi ile 18 sayısal ozellik cikarildi. Sinif dengeleme (SMOTE) ile RED/YELLOW/GREEN ornekler dengelendi. XGBoost n_estimators=100, max_depth=4, learning_rate=0.1 ile egitildi. 3-katli capraz dogrulama ile performans izlendi.'
  ),
  ...qna(
    '21. 131.677 SOCRATES kaydından ne ogrendi model?',
    'Model ozellikle su iliskileri ogrendi: dusuk iska mesafesi + yuksek gorecel hız = yuksek risk; yuksek kovaryans belirsizligi = dusuk Pc guvenilirligı; TCA suresi kısaldikca kırmızı olasılıgı artar; irtifa (GEO vs LEO) carpısma kinetik enerjisini dogrudan etkiler.'
  ),

  h2('D. SISTEM VE GELECEK SORULARI'),
  ...qna(
    '22. Sistemin eksiklikleri nelerdir?',
    'Donustuk: (1) LSTM gercek operasyonel verilerle egitilmedi — mock data kullanıldı; (2) Kovaryans tahmini ampirik, radar tabanli degil; (3) Atmosferik perturbasyon (gunes aktivitesi) detaylı modellenmedi; (4) Cok cisimli (N-body) carpısma senaryoları hesaba katılmadı. Bunlar bilimsel olarak gelismis modeller gerektiriyor.'
  ),
  ...qna(
    '23. Gelecek planları nelerdir?',
    '(1) Gercek operasyonel CDM\'lerle LSTM yeniden egitimi; (2) TUA ile resmi entegrasyon API\'si; (3) Mobil bildirim sistemi (RED alarm push notification); (4) Cok uydu ortak manevra optimizasyonu; (5) ESA Space Debris Office CDM formatıyla senkronizasyon; (6) Otomasyonlu manevra yukleme komutu (operatör onayıyla).'
  ),
  ...qna(
    '24. NASA veya ESA ile entegrasyon mumkun mu?',
    'Evet, CCSDS 508.0 CDM standardını kullandıgımız icin teknik entegrasyon mumkundur. NASA JSC\'nin CARA sistemiyle CDM paylasım protokolu kurulabilir. ESA SSA portalı da CDM kabul eder. Bir sonraki adım: TUA aracılıgıyla resmi veri paylasım anlasması.'
  ),
  ...qna(
    '25. Carpısma oldu mu hic? Test ettiniz mi?',
    'Sistemimiz gercek tarihi olaylarla test edildi: 2009 Iridium-Cosmos carpısması verisiyle geriye donuk simulasyon yapıldı (bu carpısma 1500+ kataloglı enkaz üretti). Sistem bu olayı RED olarak isaretledi. Bunun yanı sıra 41-testli otomatik test paketi (test_k2.py) ile fizik, ML ve entegrasyon testleri yapıldı.'
  ),
  ...qna(
    '26. Hangi uydular en riskli?',
    'LEO uydular (Gokturk-1, IMECE) gorecel hız ve enkaz yogunlugu nedeniyle daha sık tehditle karsilasır. GEO uydular (Turksat serisi) uzun vadede istasyon tutma gerektirir ve GEO kemeri giderek kalabalıklasıyor. IMECE en yeni ve en degerli LEO uydumuz — riskini sıkı izlemek kritik.'
  ),
  ...qna(
    '27. GEO\'daki Turksat 6A neden ozel?',
    'Turksat 6A Turk\'un ilk tamamen yerli uretim uydusudur (TUSAS). 2024\'te fırlatıldı, 14 yıl gorev omru var ve 4.229 kg kitlesiyle Turk\'un en buyuk haberlesme uydularından biridir. Bu nedenle demo senaryomuzun odagı olarak secildi — hem yeni hem kritik hem de kamu bilinciyle yuksek.'
  ),
  ...qna(
    '28. Dogrulama (validation) nasıl yapıldı?',
    'Uc kademeli dogrulama: (1) Birim testler — SGP4, CARA, Tsiolkovsky formullerinin matematiksel dogurulugu; (2) Entegrasyon testleri — K1+K2 veri akısı; (3) Gercek veri dogrulaması — Space-Track TLE ile hesaplanan konumların bilinen uydu pozisyonlarıyla karsılastırılması. 41 test her calıstırmada otomatik kontrol edilir.'
  ),
  ...qna(
    '29. Manevra simulatoru gercek mi yoksa mock mu?',
    'GERCEK fizik hesabı. Tsiolkovsky roketi denklemi uygulanıyor: yakıt kitlesi, yanma suresi, gorecel hız, yeni iska mesafesi, Pc azalması — hepsi matematiksel olarak hesaplanıyor. Uydunun gercek mass_kg ve fuel_kg degerleri kullanılıyor. Sabit kodlanmıs mock degerler kullanılmıyor.'
  ),
  ...qna(
    '30. Hidrazin neden secildi? Baska yakıtlar?',
    'Hidrazin (N2H4) GEO haberlesme uydularında standart mono-propellant. Isp ~220-230 s, basit tek bilesenli sistem, uzun depolama omru. Bi-prop (Isp ~310 s) daha verimli ama karmasık; iyon motoru (Isp 1500+) cok verimli ama dusuk itmeli (manevra suresi saatler/gunler). Turksat 6A icin hidrazin standart secimdir.'
  ),
  ...qna(
    '31. Gunes basıncı, atmosfer suruklenmesi hesaba katılıyor mu?',
    'Mevcut sistemde SGP4\'un dahili modelleri kullanılıyor — J2 terimini (Dunya yassılması) ve temel atmosfer suruklenmesini iceriyor. Gunes basıncı, yuksek mertebe yercekim terimleri (J3-J6) ve manyetik suruklenme henuz dahil edilmedi. Bu eklemeler GPS uyduları veya yuksek hassasiyet gerektiren operasyonlar icin onerilir.'
  ),
  ...qna(
    '32. Sistemin karanlık tarafı: Kotüye kullanım riski var mı?',
    'Teorik olarak bu sistem bir aktore hangi uydunun hangi zamanda nerede olacagını gosterebilir. Ancak TLE verisi zaten kamuya acık — bu bilgi herkes tarafından erisilebilir. Sistem risk azaltmaya odaklanmıstır; bu bilgi hassas operasyonel detaylar icermemektedir. TUA entegrasyonunda erisim kontrol mekanizmaları eklenecektir.'
  ),
  ...qna(
    '33. Bu sistem TUA\'ya nasıl deger katar?',
    'TUA\'ya 5 temel katki: (1) Turk uydularının bagimsız izlenmesi — Batı sistemlerine bagımlılık azalır; (2) Otonom manevra karar destegi — operatör yuku azalır; (3) CCSDS CDM ile uluslararası entegrasyon; (4) Yerli yazılım altyapısı — ulusal siber guvenlik; (5) Hackathon fikirleri kurumsal Ar-Ge\'ye donusebilir.'
  ),
  ...qna(
    '34. Proje acık kaynak olacak mı?',
    'Bu karar TUA\'ya aittir. Teknik olarak: NASA CARA, SGP4 ve CCSDS standartları acık; XGBoost ve PyTorch acık kaynak. Temel algoritmalar akademik yayınlara dayanıyor. Bir secenek: Temel katman (K1) acık, operasyonel K2-K3 kapalı kaynak modeli.'
  ),
  ...qna(
    '35. Hackathon surecinde en buyuk ogrenin neydi?',
    'En buyuk ogrenin: Gercek uzay verisinin ne kadar karmasık oldugu. SGP4 "basit" gorunse de koordinat donusumleri, GEO vs LEO farklılıkları, TLE guvenilirlik sorunları — bunların hepsi on gormedigimiz teknik zorluklardı. Bunu astıgımızda NASA CARA entegrasyonu ve oyun teorisi uygulaması cok anlasılır hale geldi.'
  ),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 12: HIZLI REFERANS
  // ════════════════════════════════════════════
  h1('BOLUM 12: HIZLI REFERANS — FORMULLER VE SAYILAR'),

  h2('Temel Formuller'),
  twoColTable([
    ['Tsiolkovsky', 'm_yakıt = m_ıslak × (1 − exp(−Δv / (Isp × 9.80665)))'],
    ['Yanma suresi', 'T = m_yakıt / (F_itmeli / (Isp × g₀))'],
    ['Manevra iyilestirme', 'Δd [km] ≈ Δv [m/s] × TCA [saat] × katsayı'],
    ['Pc sonrası', 'Pc_sonra = Pc_bası × exp(−yeni_iska / sigma)'],
    ['TLE guven', 'Guven = max(0, 100 − (TLE_yassi_saat × 1.5))'],
  ], ['Formul', 'Aciklama']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Pc Esik Degerleri'),
  twoColTable([
    ['Pc >= 1×10^-4', 'KIRMIZI — Acil manevra'],
    ['1×10^-6 <= Pc < 1×10^-4', 'SARI — Degerlendirme gerekli'],
    ['Pc < 1×10^-6', 'YESIL — Guvende, izle'],
  ], ['Esik', 'Seviye ve Eylem']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Turksat 6A Teknik Ozeti'),
  twoColTable([
    ['NORAD ID', '60233'],
    ['Fırlatma', '2024'],
    ['Kutle', '4.229 kg'],
    ['Yakit', '350 kg'],
    ['Isp', '220 s (hidrazin)'],
    ['Gorev omru', '14 yil'],
    ['Yonunde', 'GEO — 42° Dogu'],
    ['Hız', '~3.07 km/s'],
    ['Yukseklik', '35.786 km'],
  ], ['Parametre', 'Deger']),
  new Paragraph({ ...sp(60, 0) }),

  h2('Teknoloji Yigini Ozeti'),
  threeColTable([
    ['Python 3.10+', 'FastAPI, SGP4, XGBoost', 'Backend'],
    ['PyTorch', 'LSTM sinir agi', 'Yonunde tahmini'],
    ['SciPy / NumPy', 'CARA Pc entegrasyonu', 'Fizik hesabı'],
    ['React 18 + Vite', 'Kullanıcı arayuzu', 'Frontend'],
    ['CesiumJS 1.114', '3B kure gorsellestirmesi', 'Frontend'],
    ['ArcGIS World Imagery', 'Gercek uydu fotografı', 'Harita'],
    ['CCSDS 508.0', 'CDM standardi', 'Uluslararasi'],
    ['Space-Track API', 'Canli TLE verisi', 'Veri kaynagi'],
  ], ['Teknoloji', 'Kullanım', 'Katman']),
  new Paragraph({ ...sp(60, 0) }),
  pageBreak(),

  // ════════════════════════════════════════════
  // BOLUM 13: SUNUM IPUCLARI
  // ════════════════════════════════════════════
  h1('BOLUM 13: SUNUM IPUCLARI'),

  h2('Jurıyi Nasıl Etkilersin?'),
  bullet('GERCEK veriyi one cıkar: "Burada gercek Space-Track TLE\'si kullanıyoruz, simule edilmis veri degil"', true),
  bullet('Formuller sordugunda tablodan oku — ezberlemen gerekmez, referans gostermen yeterli'),
  bullet('Demo sırasında Turksat 6A\'yı sec → RED tehdit gorulecek → simulatoru ac → micro nudge sec → GREEN\'e doner'),
  bullet('"Mock" dediginde hemen "ama fizik hesabı gercek" diye devam et'),
  bullet('NASA CARA eşikleriyle ayni standartları kullandıgımızı mutlaka vurgula'),

  h2('Demo Sırasında Nelere Dikkat Et?'),
  bullet('Once API\'nin calıstıgını kontrol et (http://localhost:8000/api/status)'),
  bullet('Vite dev sunucusunun calıstıgını kontrol et (http://localhost:3000)'),
  bullet('Turksat 6A secildiginde RED tehdit gormuyorsan: "/api/refresh" cagır, onbellek yenilenir'),
  bullet('Demo modunu AC — ek RED ve YELLOW tehditler gosterilir, sunum daha etkileyici'),
  bullet('Manevra simulatorunde "Micro Nudge" sec → 12+ km iyilesme + GREEN goster → etkileyici!'),
  bullet('Kamera otomatik olarak uyduya gidecek — juri icin bekle, sonra anlatmaya basla'),

  h2('Teknik Sorulara Nasıl Yaklasırsın?'),
  bullet('"Kesin bilmiyorum ama mantıgı su sekilde..." — donustuk kabul et, anlasılır anlatim yap'),
  bullet('"Bu iyi bir soru, asıl cozumu su olurdu..." — gelecek planlarını baga'),
  bullet('Karmasık formullerde: "Temel fikir su — detaylar BOLUM X\'te" de'),
  bullet('Rakiplerle karsılastırırlarsa: guclu yanlarını one cık (ML + oyun teorisi + CDM = kombinasyon)'),

  h2('Son Tavsiye'),
  new Paragraph({
    ...sp(120, 60),
    shading: { fill: C.navy, type: ShadingType.CLEAR },
    children: [new TextRun({
      text: 'Bu rehberdeki her bolumu en az bir kez okudu isen, jurinin soracagı hicbir soru seni hazırlıksız yakalamaz. Guvenli ol, demo yaptır, formulleri referans goster. Basarılar!',
      size: 22, color: C.white, font: 'Calibri', bold: true
    })]
  }),
  new Paragraph({ ...sp(40, 40), alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: '🛰️ Turk uydularını korumak icin calisiyoruz 🇹🇷', size: 24, color: C.cyan, font: 'Calibri', bold: true })]
  }),
];

// ── Numbering ────────────────────────────────────────────────────────────────
const numbering = {
  config: [
    {
      reference: 'bullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '\u2022',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]
    },
    {
      reference: 'subbullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '\u25E6',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 360 } } }
      }]
    },
  ]
};

// ── Build doc ────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering,
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 22, color: C.black } }
    },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 36, bold: true, font: 'Calibri', color: C.navy },
        paragraph: { spacing: { before: 400, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Calibri', color: C.navy },
        paragraph: { spacing: { before: 280, after: 80 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Calibri', color: C.cyan },
        paragraph: { spacing: { before: 200, after: 60 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: C.cyan, space: 4 } },
          children: [
            new TextRun({ text: 'TUA SOPRANOS — Juri Sunumu Rehberi', size: 18, color: C.navy, font: 'Calibri', bold: true }),
            new TextRun({ text: '       Gizlilik: Takim Ici Kullanim', size: 16, color: C.dgray, font: 'Calibri' }),
          ]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: C.cyan, space: 4 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: 'Sayfa ', size: 18, color: C.dgray, font: 'Calibri' }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: C.navy, font: 'Calibri', bold: true }),
          ]
        })]
      })
    },
    children,
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('SUNUM_REHBERI.docx', buf);
  console.log('SUNUM_REHBERI.docx olusturuldu!');
}).catch(e => { console.error(e); process.exit(1); });
