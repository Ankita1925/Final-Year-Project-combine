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
    const required = ["age", "gender", "alb", "alp", "alt", "ast", "bil",
      "direct_bilirubin", "che", "chol", "crea", "ggt", "prot"];

    required.forEach((f) => {
      if (values[f] === "") errs[f] = "Required.";
    });

    const ranges = {
      age: [1, 120], alb: [0.9, 8], bil: [0.01, 80],
      direct_bilirubin: [0, 25], che: [0, 20], chol: [40, 600],
      crea: [0.01, 15], prot: [1, 12],
    };
    Object.entries(ranges).forEach(([f, [min, max]]) => {
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
      // Create the data object exactly once
      const payload = {
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
      };

      const res = await axiosInstance.post("/liver/predict", payload);
      setResult(res.data);
    } catch (err) {
      setGlobalError(err.response?.data?.message || "Submission failed.");
    } finally {
      setLoading(false);
    }
  };

  // FIXED: Moved outside of handleSubmit
  const handleDownloadPDF = async () => {
    try {
      // Create the same numeric payload used in handleSubmit
      const payload = {
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
        ascites:        values.ascites        !== "" ? parseInt(values.ascites)   : null,
        encephalopathy: values.encephalopathy !== "" ? parseInt(values.encephalopathy) : null,
      };

      // Send the payload to the /liver/report endpoint
      const response = await axiosInstance.post("/liver/report", payload, { 
        responseType: 'blob' 
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Liver_Report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("PDF Error:", err);
      alert("Failed to download PDF. Check your backend console for data type errors.");
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

  // Result data helpers
  const getDiagnosis = (r) => r?.disease || r?.primary_diagnosis || "Unknown";
  const getDecision = (r) => r?.decision || r?.recommendation || "—";
  const getCriticality = (r) => {
    if (r?.criticality) return r.criticality;
    const d = getDiagnosis(r);
    if (d === "Healthy") return "NONE";
    if (d === "Early Liver Disease") return "LOW";
    if (["Hepatitis", "Fibrosis", "Cirrhosis"].includes(d)) return "HIGH";
    return "UNKNOWN";
  };
  const getModel1Conf = (r) => r?.model1_confidence || r?.model1_probabilities || null;
  const getModel2Conf = (r) => r?.model2_confidence || r?.model2_probabilities || null;
  const isSecondaryUsed = (r) => r?.secondary_model_used ?? (r?.model2_confidence != null);
  const getSeverity = (r) => r?.severity_assessment || null;

  const criticalityColor = { NONE: "#27ae60", LOW: "#2ecc71", MEDIUM: "#f39c12", HIGH: "#e74c3c", UNKNOWN: "#95a5a6" };
  const severityColor = { "Low": "#27ae60", "Moderate": "#f39c12", "High": "#e67e22", "Very High": "#e74c3c", "Critical": "#8e1a1a" };

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
        .units-note { background: #e6f9f4; border-left: 4px solid #00d4aa; padding: 10px 14px; border-radius: 6px; font-size: 0.85rem; margin-top: 12px; color: #065f46; }

        .card { background: var(--white); border-radius: var(--radius); box-shadow: var(--shadow-md); margin-bottom: 24px; overflow: hidden; }
        .card__header { background: var(--navy); padding: 14px 22px; display: flex; align-items: center; gap: 10px; }
        .card__header h2 { font-size: 0.8rem; font-weight: 600; color: var(--white); letter-spacing: 0.08em; text-transform: uppercase; }
        .card__icon { width: 20px; height: 20px; background: var(--teal); border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; color: var(--white); flex-shrink: 0; }
        .card__body { padding: 22px 22px 20px; }

        .field-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 14px; }
        .field { display: flex; flex-direction: column; gap: 5px; border-radius: 6px; }
        .field--high { background: #fff0f0; border-left: 3px solid #e05252; padding-left: 8px; }
        .field--medium { background: #fffbea; border-left: 3px solid #f0b429; padding-left: 8px; }
        .field--low { background: var(--white); border-left: 3px solid var(--grey-200); padding-left: 8px; }
        .field label { font-size: 0.75rem; font-weight: 600; color: var(--navy); letter-spacing: 0.02em; }
        .field input, .field select { height: 40px; padding: 0 12px; border: 1.5px solid var(--grey-200); border-radius: 7px; font-size: 0.88rem; outline: none; width: 100%; transition: border-color 0.15s; }
        .field input:focus, .field select:focus { border-color: var(--teal); box-shadow: 0 0 0 3px rgba(26,122,138,0.12); }
        .field-error { font-size: 0.7rem; color: var(--red); }

        .importance-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; padding: 10px 14px; background: var(--grey-50); border-radius: 6px; border: 1px solid var(--grey-100); }
        .importance-legend span { font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 4px; color: var(--text-muted); }
        .leg--high { background: #fff0f0; border-left: 3px solid #e05252; }
        .leg--medium { background: #fffbea; border-left: 3px solid #f0b429; }
        .leg--low { background: var(--white); border: 1px solid var(--grey-200); }

        .btn-submit { width: 100%; height: 50px; background: var(--navy); color: var(--white); border: none; border-radius: var(--radius); font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: background 0.18s; margin-top: 8px; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .btn-submit:hover { background: var(--teal); }
        .btn-submit:disabled { opacity: 0.75; pointer-events: none; }
        .spinner { width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.3); border-top-color: var(--white); border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* FIXED: NEW DOWNLOAD BUTTON STYLE */
        .btn-download { background: rgba(255, 255, 255, 0.15); color: white; border: 1px solid rgba(255, 255, 255, 0.3); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: 600; transition: all 0.2s; }
        .btn-download:hover { background: rgba(255, 255, 255, 0.25); border-color: white; }

        .result-card { background: var(--white); border-radius: var(--radius); box-shadow: var(--shadow-md); margin-top: 28px; overflow: hidden; }
        .result-card__header { background: var(--navy); padding: 16px 22px; display: flex; justify-content: space-between; align-items: center; }
        .result-card__title { font-family: var(--font-serif); font-size: 1.1rem; color: var(--white); }
        .result-badge { padding: 5px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; color: var(--white); }
        .result-card__body { padding: 24px; }
        .result-disease { font-family: var(--font-serif); font-size: 1.6rem; font-weight: 700; color: var(--navy); margin-bottom: 6px; }
        .result-decision { background: var(--teal-light); border-left: 4px solid var(--teal); padding: 12px 16px; border-radius: 6px; font-size: 0.9rem; color: var(--navy); margin: 16px 0; }
        .result-decision strong { display: block; margin-bottom: 4px; font-size: 0.75rem; text-transform: uppercase; color: var(--teal); }
        
        .pipeline-pill { display: inline-block; font-size: 0.73rem; color: var(--text-muted); background: var(--grey-100); border-radius: 4px; padding: 3px 10px; margin-bottom: 16px; }
        .confidence-title { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin: 16px 0 8px; }
        .confidence-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
        .confidence-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: var(--grey-50); border-radius: 6px; font-size: 0.82rem; }
        .confidence-item span:last-child { font-weight: 600; color: var(--navy); }
        
        .severity-section { margin-top: 20px; border-top: 1px solid var(--grey-100); padding-top: 16px; }
        .severity-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        .severity-box { background: var(--grey-50); border-radius: 8px; padding: 14px 16px; border: 1px solid var(--grey-100); }
        .severity-score { font-family: var(--font-serif); font-size: 2rem; font-weight: 700; color: var(--navy); }
        .transplant-banner { background: #fdf0ee; border: 1.5px solid #e8a09a; border-radius: 8px; padding: 10px 16px; color: var(--red); font-size: 0.88rem; font-weight: 600; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
        .transplant-banner--safe { background: #eaf7ef; border-color: #a8ddb8; color: #1a7a3a; }
      `}</style>

      <header className="page-header">
        <div className="page-header__logo"><span>Liver</span></div>
        <div className="page-header__sub">Liver Disease Decision Support System</div>
      </header>

      <main className="page-content">
        <div className="form-intro">
          <h1>Patient Evaluation Form</h1>
          <p className="units-note">✅ <strong>Standard Units:</strong> Enter values in conventional clinical units (g/dL, mg/dL).</p>
        </div>

        {globalError && <div className="error-banner">{globalError}</div>}

        <form onSubmit={handleSubmit} noValidate>
          <div className="card">
            <div className="card__header"><div className="card__icon">①</div><h2>Patient Information</h2></div>
            <div className="card__body">
              <div className="field-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <div className="field field--low"><label>Age (years)</label><input type="number" value={values.age} onChange={set("age")} style={errors.age ? { borderColor: "#C0392B" } : {}} />{errors.age && <span className="field-error">{errors.age}</span>}</div>
                <div className="field field--low"><label>Biological Sex</label><select value={values.gender} onChange={set("gender")} style={errors.gender ? { borderColor: "#C0392B" } : {}}><option value="" disabled>Select…</option><option value="1">Male</option><option value="0">Female</option></select>{errors.gender && <span className="field-error">{errors.gender}</span>}</div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card__header"><div className="card__icon">②</div><h2>Core Liver Panel</h2></div>
            <div className="card__body">
              <div className="importance-legend">
                <span className="leg--high">🔴 Critical</span>
                <span className="leg--medium">🟡 Key</span>
                <span className="leg--low">⚪ Supporting</span>
              </div>
              <div className="field-grid">
                <div className="field field--medium"><label>Albumin (g/dL)</label>{inp("alb", "3.5–5.0")}{errors.alb && <span className="field-error">{errors.alb}</span>}</div>
                <div className="field field--medium"><label>ALP (U/L)</label>{inp("alp", "44–147")}{errors.alp && <span className="field-error">{errors.alp}</span>}</div>
                <div className="field field--high"><label>ALT (U/L)</label>{inp("alt", "7–56")}{errors.alt && <span className="field-error">{errors.alt}</span>}</div>
                <div className="field field--high"><label>AST (U/L)</label>{inp("ast", "10–40")}{errors.ast && <span className="field-error">{errors.ast}</span>}</div>
                <div className="field field--high"><label>Total Bilirubin (mg/dL)</label>{inp("bil", "0.2–1.2")}{errors.bil && <span className="field-error">{errors.bil}</span>}</div>
                <div className="field field--medium"><label>Direct Bilirubin (mg/dL)</label>{inp("direct_bilirubin", "0–0.3")}{errors.direct_bilirubin && <span className="field-error">{errors.direct_bilirubin}</span>}</div>
                <div className="field field--low"><label>Cholinesterase (kU/L)</label>{inp("che", "5.3–12.9")}{errors.che && <span className="field-error">{errors.che}</span>}</div>
                <div className="field field--low"><label>Cholesterol (mg/dL)</label>{inp("chol", "<200")}{errors.chol && <span className="field-error">{errors.chol}</span>}</div>
                <div className="field field--high"><label>Creatinine (mg/dL)</label>{inp("crea", "0.6–1.2")}{errors.crea && <span className="field-error">{errors.crea}</span>}</div>
                <div className="field field--medium"><label>GGT (U/L)</label>{inp("ggt", "9–48")}{errors.ggt && <span className="field-error">{errors.ggt}</span>}</div>
                <div className="field field--low"><label>Total Protein (g/dL)</label>{inp("prot", "6.0–8.3")}{errors.prot && <span className="field-error">{errors.prot}</span>}</div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card__header"><div className="card__icon">③</div><h2>Severity Scoring (Optional)</h2></div>
            <div className="card__body">
              <div className="field-grid">
                <div className="field field--high"><label>INR</label>{inp("inr", "0.8–1.1")}{errors.inr && <span className="field-error">{errors.inr}</span>}</div>
                <div className="field field--medium"><label>Sodium (mEq/L)</label>{inp("sodium", "136–145")}{errors.sodium && <span className="field-error">{errors.sodium}</span>}</div>
                <div className="field field--low"><label>Ascites</label><select value={values.ascites} onChange={set("ascites")}><option value="">Not provided</option><option value="0">0 — None</option><option value="1">1 — Mild</option><option value="2">2 — Severe</option></select></div>
                <div className="field field--low"><label>Encephalopathy</label><select value={values.encephalopathy} onChange={set("encephalopathy")}><option value="">Not provided</option><option value="0">0 — None</option><option value="1">1 — Grade 1–2</option><option value="2">2 — Grade 3–4</option></select></div>
              </div>
            </div>
          </div>

          <button type="submit" className="btn-submit" disabled={loading}>
            {loading ? <><span className="spinner" /> Running…</> : "Run Evaluation →"}
          </button>
        </form>

        {result && (() => {
          const diagnosis = getDiagnosis(result);
          const decision = getDecision(result);
          const criticality = getCriticality(result);
          const model1Conf = getModel1Conf(result);
          const model2Conf = getModel2Conf(result);
          const secondary = isSecondaryUsed(result);
          const severity = getSeverity(result);

          return (
            <div className="result-card">
              <div className="result-card__header">
                <span className="result-card__title">Evaluation Result</span>
                
                {/* FIXED: DOWNLOAD BUTTON PLACEMENT */}
                <button onClick={handleDownloadPDF} className="btn-download">⬇ Download PDF</button>

                <span className="result-badge" style={{ backgroundColor: criticalityColor[criticality] || "#95a5a6" }}>{criticality}</span>
              </div>

              <div className="result-card__body">
                <div className="result-disease">{diagnosis}</div>
                <div className="pipeline-pill">{secondary ? "Stage 1 → Stage 2 (Early assessment)" : "Stage 1 (Cirrhosis model)"}</div>
                <div className="result-decision"><strong>Recommended Action</strong>{decision}</div>

                {model1Conf && (
                  <><div className="confidence-title">Model Confidence Scores</div>
                  <div className="confidence-grid">
                    {Object.entries(model1Conf).map(([key, val]) => (
                      <div key={key} className="confidence-item"><span>{key}</span><span>{val !== null ? `${val}%` : "N/A"}</span></div>
                    ))}
                  </div></>
                )}

                {secondary && model2Conf && (
                  <><div className="confidence-title">Stage 2 Submodel Confidence</div>
                  <div className="confidence-grid">
                    {Object.entries(model2Conf).map(([key, val]) => (
                      <div key={key} className="confidence-item"><span>{key}</span><span>{val !== null ? `${val}%` : "N/A"}</span></div>
                    ))}
                  </div></>
                )}

                {severity && (
                  <div className="severity-section">
                    <h3>Severity Assessment</h3>
                    {severity.transplant_required ? <div className="transplant-banner">⚠ Transplant evaluation recommended</div> : <div className="transplant-banner transplant-banner--safe">✓ Medical management appropriate</div>}
                    <div className="severity-grid">
                      <div className="severity-box"><h4>MELD Score</h4><div className="severity-score">{severity.meld_score}</div><div style={{ color: severityColor[severity.meld?.risk_level] || "inherit" }}>{severity.meld?.risk_level} Risk</div><div className="severity-desc">{severity.meld?.description}</div></div>
                      <div className="severity-box"><h4>Child-Pugh</h4><div className="severity-score">{severity.child_pugh?.score}</div><div>{severity.child_pugh?.classification}</div><div className="severity-desc">{severity.child_pugh?.description}</div></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })()}
      </main>
    </>
  );
};

export default Liver;