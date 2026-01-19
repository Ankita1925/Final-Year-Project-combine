////////////////////////////////////////////////////////////////////
//
// File Name : Lung.jsx
// Description : Lung disease prediction form and result display
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { useState } from "react";
import axiosInstance from "../../api/axiosInstance";
import Navbar from "../../components/Navbar";

function Lung() {
  /**
   * ////////////////////////////////////////////////////////////////
   *
   * Function Name : Lung
   * Description   : Collects lung parameters and calls prediction API
   * Author        : Pradhumnya Changdev Kalsait
   * Date          : 17/01/26
   *
   * ////////////////////////////////////////////////////////////////
   */

  const [age, setAge] = useState("");
  const [smokingYears, setSmokingYears] = useState("");
  const [spo2, setSpo2] = useState("");

  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setErrorMessage("");
    setPredictionResult(null);
    setLoading(true);

    try {
      const response = await axiosInstance.post("/lung/predict", {
        age: Number(age),
        smoking_years: Number(smokingYears),
        spo2: Number(spo2),
      });

      setPredictionResult(response.data);
    } catch (error) {
      setErrorMessage("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 flex justify-center items-start p-10">
        <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-lg">
          <h2 className="text-2xl font-bold mb-6">
            Lung Disease Prediction
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium">
                Age
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border rounded"
                value={age}
                onChange={(event) => setAge(event.target.value)}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium">
                Smoking Years
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border rounded"
                value={smokingYears}
                onChange={(event) => setSmokingYears(event.target.value)}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium">
                SpO₂ (%)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border rounded"
                value={spo2}
                onChange={(event) => setSpo2(event.target.value)}
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
            >
              {loading ? "Predicting..." : "Predict"}
            </button>
          </form>

          {errorMessage && (
            <p className="text-red-600 mt-4">
              {errorMessage}
            </p>
          )}

          {predictionResult && (
            <div className="mt-6 bg-gray-50 p-4 rounded border">
              <h3 className="text-lg font-semibold mb-2">
                Prediction Result
              </h3>

              <p>
                <strong>Disease:</strong>{" "}
                {predictionResult.disease}
              </p>

              <p>
                <strong>Criticality:</strong>{" "}
                {predictionResult.criticality}
              </p>

              <p>
                <strong>Decision:</strong>{" "}
                {predictionResult.decision}
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default Lung;
