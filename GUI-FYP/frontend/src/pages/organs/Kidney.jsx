import React, { useState } from "react";

function Kidney() {
  const [formData, setFormData] = useState({
    serum_creatinine: "",
    gfr: "",
    bun: "",
    serum_calcium: "",
    ana: "",
    c3_c4: "",
    hematuria: "",
    oxalate_levels: "",
    urine_ph: "",
    blood_pressure: "",
    physical_activity: "",
    diet: "",
    water_intake: "",
    smoking: "",
    alcohol: "",
    painkiller_usage: "",
    family_history: "",
    weight_changes: "",
    stress_level: "",
    months: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Map categorical strings to numerical codes expected by backend
  const encodeData = (data) => {
    const mapYesNo = { yes: 1, no: 0 };
    const mapNormalLow = { normal: 0, low: 1 };
    const mapPhysicalActivity = { daily: 2, weekly: 1, rarely: 0 };
    const mapDiet = { "high protein": 2, balanced: 1, low: 0 };
    const mapSmokingAlcoholPainkiller = { daily: 2, occasional: 1, no: 0 };
    const mapWeightChanges = { stable: 0, loss: 1, gain: 2 };
    const mapStressLevel = { low: 0, moderate: 1, high: 2 };

    return {
      ...data,

      // Numeric fields parsed to float or int
      serum_creatinine: parseFloat(data.serum_creatinine) || 0,
      gfr: parseFloat(data.gfr) || 0,
      bun: parseFloat(data.bun) || 0,
      serum_calcium: parseFloat(data.serum_calcium) || 0,
      oxalate_levels: parseFloat(data.oxalate_levels) || 0,
      urine_ph: parseFloat(data.urine_ph) || 0,
      blood_pressure: parseFloat(data.blood_pressure) || 0,
      months: parseInt(data.months) || 0,

      // Encoded categorical fields
      ana: mapYesNo[data.ana.toLowerCase()] ?? 0,
      c3_c4: mapNormalLow[data.c3_c4.toLowerCase()] ?? 0,
      hematuria: mapYesNo[data.hematuria.toLowerCase()] ?? 0,

      physical_activity:
        mapPhysicalActivity[data.physical_activity.toLowerCase()] ?? 0,
      diet: mapDiet[data.diet.toLowerCase()] ?? 0,
      water_intake: parseFloat(data.water_intake) || 0,
      smoking: mapSmokingAlcoholPainkiller[data.smoking.toLowerCase()] ?? 0,
      alcohol: mapSmokingAlcoholPainkiller[data.alcohol.toLowerCase()] ?? 0,
      painkiller_usage:
        mapSmokingAlcoholPainkiller[data.painkiller_usage.toLowerCase()] ?? 0,
      family_history: mapYesNo[data.family_history.toLowerCase()] ?? 0,
      weight_changes: mapWeightChanges[data.weight_changes.toLowerCase()] ?? 0,
      stress_level: mapStressLevel[data.stress_level.toLowerCase()] ?? 0,
    };
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const encodedData = encodeData(formData);

      const response = await fetch("http://127.0.0.1:5000/api/kidney/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(encodedData),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Error connecting to backend: " + error.message);
    }

    setLoading(false);
  };

const Input = ({ label, name, type = "text" }) => (
  <div className="flex flex-col">
    <label className="text-sm font-medium text-gray-700 mb-1">{label}</label>
    <input
      type={type}
      name={name}
      value={formData[name] ?? ""}
      onChange={handleChange}
      required
      className="px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      autoComplete="off"
    />
  </div>
);

const Select = ({ label, name, options }) => (
  <div className="flex flex-col">
    <label className="text-sm font-medium text-gray-700 mb-1">{label}</label>
    <select
      name={name}
      value={formData[name] ?? ""}
      onChange={handleChange}
      required
      className="px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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


  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-center text-gray-800">
          Kidney Disease Prediction
        </h2>
        <p className="text-center text-gray-500 mt-2 mb-8">
          Enter patient clinical data to predict kidney disease stage
        </p>

        <form
          onSubmit={handleSubmit}
          className="bg-white p-8 rounded-xl shadow-lg space-y-8"
        >
          {/* Lab Values */}
          <div>
            <h3 className="text-lg font-semibold text-blue-600 mb-4 border-b pb-1">
              Blood & Lab Values
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <Input label="Serum Creatinine (mg/dL)" name="serum_creatinine" />
              <Input label="GFR (mL/min)" name="gfr" />
              <Input label="Blood Urea Nitrogen" name="bun" />
              <Input label="Serum Calcium" name="serum_calcium" />
              <Input label="Oxalate Levels" name="oxalate_levels" />
              <Input label="Urine pH" name="urine_ph" />
              <Input label="Blood Pressure" name="blood_pressure" />
              <Input label="Water Intake (Liters)" name="water_intake" />
              <Input label="Months" name="months" />
            </div>
          </div>

          {/* Clinical Conditions */}
          <div>
            <h3 className="text-lg font-semibold text-blue-600 mb-4 border-b pb-1">
              Clinical Conditions
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <Select label="Anemia" name="ana" options={["yes", "no"]} />
              <Select label="C3/C4 Levels" name="c3_c4" options={["normal", "low"]} />
              <Select label="Hematuria" name="hematuria" options={["yes", "no"]} />
              <Select
                label="Physical Activity"
                name="physical_activity"
                options={["daily", "weekly", "rarely"]}
              />
              <Select
                label="Diet"
                name="diet"
                options={["high protein", "balanced", "low"]}
              />
              <Select label="Smoking" name="smoking" options={["daily", "occasional", "no"]} />
              <Select label="Alcohol" name="alcohol" options={["daily", "occasional", "no"]} />
              <Select
                label="Painkiller Usage"
                name="painkiller_usage"
                options={["daily", "occasional", "no"]}
              />
              <Select label="Family History" name="family_history" options={["yes", "no"]} />
              <Select
                label="Weight Changes"
                name="weight_changes"
                options={["stable", "loss", "gain"]}
              />
              <Select
                label="Stress Level"
                name="stress_level"
                options={["low", "moderate", "high"]}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-blue-600 text-white rounded-lg font-semibold text-lg hover:bg-blue-700 transition disabled:bg-gray-400"
          >
            {loading ? "Predicting..." : "Predict Disease"}
          </button>
        </form>

        {/* Result */}
        {result && (
          <div className="mt-8 bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4 text-gray-800">
              Prediction Result
            </h3>
            <p>
              <b>Organ:</b> {result.organ}
            </p>
            <p>
              <b>Disease:</b> {result.disease}
            </p>
            <p>
              <b>Criticality:</b> {result.criticality}
            </p>
            <p>
              <b>Decision:</b> {result.decision}
            </p>
            <p>
              <b>Confidence:</b> {result.confidence}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Kidney;
