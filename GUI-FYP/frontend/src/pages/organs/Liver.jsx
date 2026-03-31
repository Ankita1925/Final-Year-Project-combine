import React, { useState } from "react";
import axiosInstance from "../../api/axiosInstance";

const Liver = () => {
  const [values, setValues] = useState({
    age: "", gender: "",
    alb: "", alp: "", alt: "", ast: "", bil: "", direct_bilirubin: "",
    che: "", chol: "", crea: "", ggt: "", prot: "",
    inr: "", sodium: "", ascites: "", encephalopathy: "",
  });
  const [errors, setErrors] = useState({});
  const [globalError, setGlobalError] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const set = (field) => (e) => {
    setValues((v) => ({ ...v, [field]: e.target.value }));
    setErrors((er) => ({ ...er, [field]: "" }));
    setGlobalError("");
  };

  const validate = () => {
    const errs = {};
    const required = ["age","gender","alb","alp","alt","ast","bil",
      "direct_bilirubin","che","chol","crea","ggt","prot"];

    required.forEach((f) => {
      if (values[f] === "") errs[f] = "Required.";
    });

    const ranges = {
      age:[1,120], alb:[0.9,8], bil:[0.01,80],
      direct_bilirubin:[0,25], che:[0,20], chol:[40,600],
      crea:[0.01,15], prot:[1,12],
    };
    Object.entries(ranges).forEach(([f,[min,max]]) => {
      const v = parseFloat(values[f]);
      if (!isNaN(v) && (v < min || v > max))
        errs[f] = `Must be ${min}–${max}.`;
    });

    if (values.inr !== "" && parseFloat(values.inr) < 0.01)
      errs.inr = "Must be ≥ 0.01.";
    if (values.sodium !== "") {
      const s = parseFloat(values.sodium);
      if (s < 100 || s > 180) errs.sodium = "Must be 100–180.";
    }
    return errs;
  };

  // ✅ FIX 1: uses axiosInstance (sends JWT token automatically)
  const handleSubmit = async (e) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) {
      setErrors(errs);
      setGlobalError("Please correct the highlighted fields before submitting.");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await axiosInstance.post("/liver/predict", {
        age:              parseFloat(values.age),
        gender:           parseInt(values.gender),
        alb:              parseFloat(values.alb),
        alp:              parseFloat(values.alp),
        alt:              parseFloat(values.alt),
        ast:              parseFloat(values.ast),
        bil:              parseFloat(values.bil),
        direct_bilirubin: parseFloat(values.direct_bilirubin),
        che:              parseFloat(values.che),
        chol:             parseFloat(values.chol),
        crea:             parseFloat(values.crea),
        ggt:              parseFloat(values.ggt),
        prot:             parseFloat(values.prot),
        inr:            values.inr            ? parseFloat(values.inr)            : null,
        sodium:         values.sodium         ? parseFloat(values.sodium)         : null,
        ascites:        values.ascites        ? parseInt(values.ascites)          : null,
        encephalopathy: values.encephalopathy ? parseInt(values.encephalopathy)   : null,
      });
      // ✅ FIX 2: saves result to state
      setResult(res.data);
    } catch (err) {
      setGlobalError(err.response?.data?.message || "Submission failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const inp = (field, placeholder, step = "0.01") => (
    <input
      type="number"
      name={field}
      value={values[field]}
      onChange={set(field)}
      placeholder={placeholder}
      step={step}
      style={errors[field] ? { borderColor: "#C0392B", boxShadow: "0 0 0 3px rgba(192,57,43,0.10)" } : {}}
    />
  );

  const criticalityColor = {
    NONE:    "#27ae60",
    LOW:     "#2ecc71",
    MEDIUM:  "#f39c12",
    HIGH:    "#e74c3c",
    UNKNOWN: "#95a5a6",
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
        :root {
          --navy: #1B3A5C; --teal: #1A7A8A; --teal-light: #E8F4F6;
          --teal-mid: #A8D4DA; --red: #C0392B; --grey-50: #F7F9FB;
          --grey-100: #EEF1F5; --grey-200: #D8DDE6; --text: #1E2A38;
          --text-muted: #5A6A7A; --white: #FFFFFF; --radius: 10px;
          --shadow-md: 0 4px 16px rgba(27,58,92,0.10);
          --font-serif: 'Source Serif 4', Georgia, serif;
          --font-sans: 'DM Sans', system-ui, sans-serif;
        }
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: var(--font-sans); background: var(--grey-50); color: var(--text); }
        .page-header { background: var(--navy); padding: 28px 40px 24px; border-bottom: 3px solid var(--teal); }
        .page-header__logo { font-family: var(--font-serif); font-size: 1.75rem; font-weight: 700; color: var(--white); }
        .page-header__logo span { color: var(--teal-mid); }
        .page-header__sub { font-size: 0.8rem; color: #8BAFC8; margin-top: 2px; font-weight: 300; letter-spacing: 0.04em; text-transform: uppercase; }
        .page-content { max-width: 900px; margin: 0 auto; padding: 36px 24px 60px; }
        .form-intro { margin-bottom: 28px; }
        .form-intro h1 { font-family: var(--font-serif); font-size: 1.55rem; font-weight: 600; color: var(--navy); }
        .form-intro p { font-size: 0.88rem; color: var(--text-muted); margin-top: 6px; }
        .units-note { background: #e6f9f4; border-left: 4px solid #00d4aa; padding: 10px 14px; border-radius: 6px; font-size: 0.85rem; margin-top: 12px; color: #065f46; }
        .card { background: var(--white); border-radius: var(--radius); box-shadow: var(--shadow-md); margin-bottom: 24px; overflow: hidden; }
        .card__header { background: var(--navy); padding: 14px 22px; display: flex; align-items: center; gap: 10px; }
        .card__header h2 { font-size: 0.8rem; font-weight: 600; color: var(--white); letter-spacing: 0.08em; text-transform: uppercase; }
        .card__icon { width: 20px; height: 20px; background: var(--teal); border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; color: var(--white); flex-shrink: 0; }
        .card__body { padding: 22px 22px 20px; }
        .field-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 14px; }
        .field-grid--2col { grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); }
        .field { display: flex; flex-direction: column; gap: 5px; border-radius: 6px; }
        .field--high   { background: #fff0f0; border-left: 3px solid #e05252; padding-left: 8px; }
        .field--medium { background: #fffbea; border-left: 3px solid #f0b429; padding-left: 8px; }
        .field--low    { background: var(--white); border-left: 3px solid var(--grey-200); padding-left: 8px; }
        .field label { font-size: 0.75rem; font-weight: 600; color: var(--navy); letter-spacing: 0.02em; }
        .field label .unit { font-weight: 400; color: var(--text-muted); margin-left: 4px; }
        .field label .optional { font-weight: 400; color: var(--teal); margin-left: 4px; font-style: italic; }
        .field input, .field select { height: 40px; padding: 0 12px; border: 1.5px solid var(--grey-200); border-radius: 7px; font-family: var(--font-sans); font-size: 0.88rem; color: var(--text); background: var(--white); outline: none; width: 100%; transition: border-color 0.15s, box-shadow 0.15s; }
        .field input:focus, .field select:focus { border-color: var(--teal); box-shadow: 0 0 0 3px rgba(26,122,138,0.12); }
        .field-hint  { font-size: 0.7rem; color: var(--text-muted); }
        .field-error { font-size: 0.7rem; color: var(--red); }
        .importance-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; padding: 10px 14px; background: var(--grey-50); border-radius: 6px; border: 1px solid var(--grey-100); }
        .importance-legend span { font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 4px; color: var(--text-muted); }
        .leg--high   { background: #fff0f0; border-left: 3px solid #e05252; }
        .leg--medium { background: #fffbea; border-left: 3px solid #f0b429; }
        .leg--low    { background: var(--white); border: 1px solid var(--grey-200); }
        .optional-note { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 14px; padding: 8px 12px; background: var(--teal-light); border-left: 3px solid var(--teal); border-radius: 4px; }
        .error-banner { background: #FDF0EE; border: 1.5px solid #E8A09A; border-radius: 8px; padding: 14px 18px; color: var(--red); font-size: 0.88rem; margin-bottom: 20px; }
        .btn-submit { width: 100%; height: 50px; background: var(--navy); color: var(--white); border: none; border-radius: var(--radius); font-family: var(--font-sans); font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: background 0.18s; margin-top: 8px; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .btn-submit:hover { background: var(--teal); }
        .btn-submit:disabled { opacity: 0.75; pointer-events: none; }
        .spinner { width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.3); border-top-color: var(--white); border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result-card { background: var(--white); border-radius: var(--radius); box-shadow: var(--shadow-md); margin-top: 28px; overflow: hidden; }
        .result-card__header { background: var(--navy); padding: 16px 22px; display: flex; justify-content: space-between; align-items: center; }
        .result-card__title { font-family: var(--font-serif); font-size: 1.1rem; color: var(--white); }
        .result-badge { padding: 5px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; color: var(--white); letter-spacing: 0.05em; }
        .result-card__body { padding: 24px; }
        .result-disease { font-family: var(--font-serif); font-size: 1.6rem; font-weight: 700; color: var(--navy); margin-bottom: 6px; }
        .result-decision { background: var(--teal-light); border-left: 4px solid var(--teal); padding: 12px 16px; border-radius: 6px; font-size: 0.9rem; color: var(--navy); margin: 16px 0; }
        .result-decision strong { display: block; margin-bottom: 4px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--teal); }
        .confidence-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; margin-top: 8px; }
        .confidence-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: var(--grey-50); border-radius: 6px; font-size: 0.82rem; }
        .confidence-item span:last-child { font-weight: 600; color: var(--navy); }
        .confidence-title { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin: 16px 0 8px; }
        @media (max-width: 600px) {
          .page-header { padding: 20px; }
          .page-content { padding: 20px 14px 40px; }
          .field-grid { grid-template-columns: 1fr 1fr; }
        }
      `}</style>

      <header className="page-header">
        <div className="page-header__logo"><span>Liver</span></div>
        <div className="page-header__sub">Liver Disease Decision Support System</div>
      </header>

      <main className="page-content">
        <div className="form-intro">
          <h1>Patient Evaluation Form</h1>
          <p>Enter the patient's laboratory values below. All required fields must be completed.</p>
          <p className="units-note">✅ <strong>Standard Units:</strong> Enter all values in conventional clinical units (g/dL, mg/dL). Unit conversions are handled automatically in the backend.</p>
        </div>

        {globalError && <div className="error-banner">{globalError}</div>}

        <form onSubmit={handleSubmit} noValidate>

          <div className="card">
            <div className="card__header">
              <div className="card__icon">①</div>
              <h2>Patient Information</h2>
            </div>
            <div className="card__body">
              <div className="field-grid field-grid--2col">
                <div className="field field--low">
                  <label>Age <span className="unit">(years)</span></label>
                  <input type="number" name="age" value={values.age} onChange={set("age")} placeholder="e.g. 45" min="1" max="120" step="1" style={errors.age ? { borderColor: "#C0392B" } : {}} />
                  {errors.age && <span className="field-error">{errors.age}</span>}
                </div>
                <div className="field field--low">
                  <label>Biological Sex</label>
                  <select name="gender" value={values.gender} onChange={set("gender")} style={errors.gender ? { borderColor: "#C0392B" } : {}}>
                    <option value="" disabled>Select…</option>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                  </select>
                  {errors.gender && <span className="field-error">{errors.gender}</span>}
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card__header">
              <div className="card__icon">②</div>
              <h2>Core Liver Panel</h2>
            </div>
            <div className="card__body">
              <div className="importance-legend">
                <span className="leg--high">🔴 Critical</span>
                <span className="leg--medium">🟡 Key</span>
                <span className="leg--low">⚪ Supporting</span>
              </div>
              <div className="field-grid">
                <div className="field field--medium"><label>Albumin <span className="unit">(g/dL)</span></label>{inp("alb","Normal: 3.5–5.0")}<span className="field-hint">3.5–5.0 g/dL</span>{errors.alb&&<span className="field-error">{errors.alb}</span>}</div>
                <div className="field field--medium"><label>ALP <span className="unit">(U/L)</span></label>{inp("alp","Normal: 44–147","0.1")}<span className="field-hint">44–147 U/L</span>{errors.alp&&<span className="field-error">{errors.alp}</span>}</div>
                <div className="field field--high"><label>ALT <span className="unit">(U/L)</span></label>{inp("alt","Normal: 7–56","0.1")}<span className="field-hint">7–56 U/L</span>{errors.alt&&<span className="field-error">{errors.alt}</span>}</div>
                <div className="field field--high"><label>AST <span className="unit">(U/L)</span></label>{inp("ast","Normal: 10–40","0.1")}<span className="field-hint">10–40 U/L</span>{errors.ast&&<span className="field-error">{errors.ast}</span>}</div>
                <div className="field field--high"><label>Total Bilirubin <span className="unit">(mg/dL)</span></label>{inp("bil","Normal: 0.2–1.2")}<span className="field-hint">0.2–1.2 mg/dL</span>{errors.bil&&<span className="field-error">{errors.bil}</span>}</div>
                <div className="field field--medium"><label>Direct Bilirubin <span className="unit">(mg/dL)</span></label>{inp("direct_bilirubin","Normal: 0–0.3")}<span className="field-hint">0–0.3 mg/dL</span>{errors.direct_bilirubin&&<span className="field-error">{errors.direct_bilirubin}</span>}</div>
                <div className="field field--low"><label>Cholinesterase <span className="unit">(kU/L)</span></label>{inp("che","Normal: 5.3–12.9")}<span className="field-hint">5.3–12.9 kU/L</span>{errors.che&&<span className="field-error">{errors.che}</span>}</div>
                <div className="field field--low"><label>Cholesterol <span className="unit">(mg/dL)</span></label>{inp("chol","Normal: <200","0.1")}<span className="field-hint">&lt;200 mg/dL</span>{errors.chol&&<span className="field-error">{errors.chol}</span>}</div>
                <div className="field field--high"><label>Creatinine <span className="unit">(mg/dL)</span></label>{inp("crea","Normal: 0.6–1.2")}<span className="field-hint">0.6–1.2 mg/dL</span>{errors.crea&&<span className="field-error">{errors.crea}</span>}</div>
                <div className="field field--medium"><label>GGT <span className="unit">(U/L)</span></label>{inp("ggt","Normal: 9–48","0.1")}<span className="field-hint">9–48 U/L</span>{errors.ggt&&<span className="field-error">{errors.ggt}</span>}</div>
                <div className="field field--low"><label>Total Protein <span className="unit">(g/dL)</span></label>{inp("prot","Normal: 6.0–8.3")}<span className="field-hint">6.0–8.3 g/dL</span>{errors.prot&&<span className="field-error">{errors.prot}</span>}</div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card__header">
              <div className="card__icon">③</div>
              <h2>Severity Scoring <span style={{ fontWeight: 300, textTransform: "none", fontSize: "0.85em" }}>(Optional — MELD &amp; Child-Pugh)</span></h2>
            </div>
            <div className="card__body">
              <p className="optional-note">Optional but recommended if cirrhosis is suspected. INR + Bilirubin + Creatinine needed for MELD. Ascites + Encephalopathy needed for Child-Pugh.</p>
              <div className="field-grid">
                <div className="field field--high"><label>INR <span className="unit">(ratio)</span> <span className="optional">optional</span></label>{inp("inr","Normal: 0.8–1.1")}{errors.inr&&<span className="field-error">{errors.inr}</span>}</div>
                <div className="field field--medium"><label>Sodium <span className="unit">(mEq/L)</span> <span className="optional">optional</span></label>{inp("sodium","Normal: 136–145","0.1")}{errors.sodium&&<span className="field-error">{errors.sodium}</span>}</div>
                <div className="field field--low">
                  <label>Ascites <span className="optional">optional</span></label>
                  <select name="ascites" value={values.ascites} onChange={set("ascites")}>
                    <option value="">Not provided</option>
                    <option value="0">0 — None</option>
                    <option value="1">1 — Mild</option>
                    <option value="2">2 — Severe</option>
                  </select>
                </div>
                <div className="field field--low">
                  <label>Encephalopathy <span className="optional">optional</span></label>
                  <select name="encephalopathy" value={values.encephalopathy} onChange={set("encephalopathy")}>
                    <option value="">Not provided</option>
                    <option value="0">0 — None</option>
                    <option value="1">1 — Grade 1–2</option>
                    <option value="2">2 — Grade 3–4</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          <button type="submit" className="btn-submit" disabled={loading}>
            {loading ? <><span className="spinner" /> Running…</> : "Run Evaluation →"}
          </button>

        </form>

        {/* ✅ FIX 3: Result display */}
        {result && (
          <div className="result-card">
            <div className="result-card__header">
              <span className="result-card__title">Evaluation Result</span>
              <span className="result-badge" style={{ backgroundColor: criticalityColor[result.criticality] || "#666" }}>
                {result.criticality}
              </span>
            </div>
            <div className="result-card__body">
              <div className="result-disease">{result.disease}</div>
              <div className="result-decision">
                <strong>Recommended Action</strong>
                {result.decision}
              </div>
              {result.model1_confidence && (
                <>
                  <div className="confidence-title">Model Confidence Scores</div>
                  <div className="confidence-grid">
                    {Object.entries(result.model1_confidence).map(([key, val]) => (
                      <div key={key} className="confidence-item">
                        <span>{key}</span>
                        <span>{val !== null ? val + "%" : "N/A"}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
              {result.model2_confidence && (
                <>
                  <div className="confidence-title">Secondary Model Scores</div>
                  <div className="confidence-grid">
                    {Object.entries(result.model2_confidence).map(([key, val]) => (
                      <div key={key} className="confidence-item">
                        <span>{key}</span>
                        <span>{val !== null ? val + "%" : "N/A"}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        )}

      </main>
    </>
  );
};

export default Liver;