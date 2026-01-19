////////////////////////////////////////////////////////////////////
 //
// File Name : Kidney.jsx
 // Description : Kidney disease prediction UI using ML pipeline
 // Author      : Pradhumnya Changdev Kalsait
 // Date        : 19/01/26
 //
 ////////////////////////////////////////////////////////////////////

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Droplet,
  HeartPulse,
  Brain,
  Loader2,
} from "lucide-react";
import axiosInstance from "../../api/axiosInstance";
import Navbar from "../../components/Navbar";

/* ================= Reusable Inputs ================= */

function Input({ label, name, value, onChange }) {
  return (
    <div>
      <label className="block text-sm font-medium mb-1">{label}</label>
      <input
        type="number"
        step="any"
        name={name}
        value={value}
        onChange={onChange}
        required
        className="w-full border px-3 py-2 rounded-lg focus:ring-2 focus:ring-blue-500"
      />
    </div>
  );
}

function Select({ label, name, value, onChange, options }) {
  return (
    <div>
      <label className="block text-sm font-medium mb-1">{label}</label>
      <select
        name={name}
        value={value}
        onChange={onChange}
        required
        className="w-full border px-3 py-2 rounded-lg focus:ring-2 focus:ring-blue-500"
      >
        <option value="">Select</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

/* ================= Main Component ================= */

function Kidney() {
  const [formData, setFormData] = useState({
    serum_creatinine: "",
    gfr: "",
    bun: "",
    serum_calcium: "",
    c3_c4: "",
    oxalate_levels: "",
    urine_ph: "",
    blood_pressure: "",
    water_intake: "",
    months: "",

    ana: "",
    hematuria: "",
    painkiller_usage: "",
    family_history: "",

    physical_activity: "",
    diet: "",
    smoking: "",
    alcohol: "",
    weight_changes: "",
    stress_level: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleChange(e) {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        serum_creatinine: Number(formData.serum_creatinine),
        gfr: Number(formData.gfr),
        bun: Number(formData.bun),
        serum_calcium: Number(formData.serum_calcium),
        c3_c4: Number(formData.c3_c4),
        oxalate_levels: Number(formData.oxalate_levels),
        urine_ph: Number(formData.urine_ph),
        blood_pressure: Number(formData.blood_pressure),
        water_intake: Number(formData.water_intake),
        months: Number(formData.months),

        ana: formData.ana === "yes" ? 1 : 0,
        hematuria: formData.hematuria === "yes" ? 1 : 0,
        painkiller_usage: formData.painkiller_usage === "yes" ? 1 : 0,
        family_history: formData.family_history === "yes" ? 1 : 0,

        physical_activity: formData.physical_activity,
        diet: formData.diet,
        smoking: formData.smoking,
        alcohol: formData.alcohol,
        weight_changes: formData.weight_changes,
        stress_level: formData.stress_level,
      };

      const response = await axiosInstance.post(
        "/kidney/predict",
        payload
      );

      setResult(response.data);
    } catch (error) {
      alert("Prediction failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Navbar />

      <div className="relative min-h-screen overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 z-[-1] bg-white bg-[radial-gradient(100%_60%_at_50%_0%,rgba(0,163,255,0.15)_0,rgba(0,163,255,0)_60%,rgba(0,163,255,0)_100%)]" />

        <div className="max-w-7xl mx-auto px-6 py-14">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-10"
          >
            <h2 className="text-4xl font-extrabold">
              Kidney Disease Prediction
            </h2>
            <p className="mt-3 text-gray-600 text-lg">
              Enter patient clinical parameters for AI-based CKD analysis
            </p>
          </motion.div>

          {/* Form */}
          <motion.form
            onSubmit={handleSubmit}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl p-8 space-y-10"
          >
            {/* ================= NUMERIC ================= */}
            <section>
              <h3 className="flex items-center gap-2 text-xl font-semibold mb-4">
                <Activity /> Clinical Measurements
              </h3>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <Input label="Serum Creatinine" name="serum_creatinine" value={formData.serum_creatinine} onChange={handleChange} />
                <Input label="GFR" name="gfr" value={formData.gfr} onChange={handleChange} />
                <Input label="Blood Urea Nitrogen" name="bun" value={formData.bun} onChange={handleChange} />
                <Input label="Serum Calcium" name="serum_calcium" value={formData.serum_calcium} onChange={handleChange} />
                <Input label="C3/C4 Levels" name="c3_c4" value={formData.c3_c4} onChange={handleChange} />
                <Input label="Oxalate Levels" name="oxalate_levels" value={formData.oxalate_levels} onChange={handleChange} />
                <Input label="Urine pH" name="urine_ph" value={formData.urine_ph} onChange={handleChange} />
                <Input label="Blood Pressure" name="blood_pressure" value={formData.blood_pressure} onChange={handleChange} />
                <Input label="Water Intake (L)" name="water_intake" value={formData.water_intake} onChange={handleChange} />
                <Input label="Months" name="months" value={formData.months} onChange={handleChange} />
              </div>
            </section>

            {/* ================= BINARY ================= */}
            <section>
              <h3 className="flex items-center gap-2 text-xl font-semibold mb-4">
                <HeartPulse /> Medical History
              </h3>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <Select label="Anemia" name="ana" value={formData.ana} onChange={handleChange} options={["yes", "no"]} />
                <Select label="Hematuria" name="hematuria" value={formData.hematuria} onChange={handleChange} options={["yes", "no"]} />
                <Select label="Painkiller Usage" name="painkiller_usage" value={formData.painkiller_usage} onChange={handleChange} options={["yes", "no"]} />
                <Select label="Family History" name="family_history" value={formData.family_history} onChange={handleChange} options={["yes", "no"]} />
              </div>
            </section>

            {/* ================= LIFESTYLE ================= */}
            <section>
              <h3 className="flex items-center gap-2 text-xl font-semibold mb-4">
                <Brain /> Lifestyle Factors
              </h3>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <Select label="Physical Activity" name="physical_activity" value={formData.physical_activity} onChange={handleChange} options={["daily", "weekly", "rarely"]} />
                <Select label="Diet" name="diet" value={formData.diet} onChange={handleChange} options={["high protein", "balanced", "low salt"]} />
                <Select label="Smoking" name="smoking" value={formData.smoking} onChange={handleChange} options={["yes", "no"]} />
                <Select label="Alcohol" name="alcohol" value={formData.alcohol} onChange={handleChange} options={["daily", "occasionally", "never"]} />
                <Select label="Weight Changes" name="weight_changes" value={formData.weight_changes} onChange={handleChange} options={["stable", "loss", "gain"]} />
                <Select label="Stress Level" name="stress_level" value={formData.stress_level} onChange={handleChange} options={["low", "moderate", "high"]} />
              </div>
            </section>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition disabled:opacity-60"
            >
              {loading ? <Loader2 className="animate-spin" /> : <Droplet />}
              {loading ? "Predicting..." : "Predict Kidney Disease"}
            </button>
          </motion.form>

          {/* ================= RESULT ================= */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-10 bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl p-8"
            >
              <h3 className="text-2xl font-semibold mb-6">
                Prediction Result
              </h3>

              <div className="grid gap-3 text-lg">
                <p><strong>Stage:</strong> {result.stage}</p>
                <p><strong>Disease:</strong> {result.disease}</p>
                <p><strong>Criticality:</strong> {result.criticality}</p>
                <p><strong>Decision:</strong> {result.decision}</p>
                <p><strong>Confidence:</strong> {result.confidence}</p>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </>
  );
}

export default Kidney;
