////////////////////////////////////////////////////////////////////
 //
// File Name : Lung.jsx
 // Description : Lung disease prediction using chest X-ray image
 // Author      : Pradhumnya Changdev Kalsait
 // Date        : 18/01/26
 //
 ////////////////////////////////////////////////////////////////////

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud, ImageIcon, Loader2 } from "lucide-react";
import axiosInstance from "../../api/axiosInstance";
import Navbar from "../../components/Navbar";

function Lung() {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  /* ================= FILE HANDLING ================= */
  function handleFileChange(event) {
    const file = event.target.files[0];
    if (!file) return;

    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setErrorMessage("");
  }

  useEffect(() => {
    return () => previewUrl && URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  /* ================= SUBMIT ================= */
  async function handleSubmit(event) {
    event.preventDefault();

    if (!imageFile) {
      setErrorMessage("Please upload a chest X-ray image.");
      return;
    }

    setLoading(true);
    setResult(null);
    setErrorMessage("");

    try {
      const formData = new FormData();
      formData.append("image", imageFile);

      const response = await axiosInstance.post(
        "/lung/predict",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(response.data);
    } catch (error) {
      if (error.response?.status === 403) {
        setErrorMessage("Doctor access only.");
      } else if (error.response?.data?.error) {
        setErrorMessage(error.response.data.error);
      } else {
        setErrorMessage("Prediction failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  /* ================= BADGE ================= */
  function getBadge(level) {
    const map = {
      LOW: "bg-green-100 text-green-700",
      MEDIUM: "bg-yellow-100 text-yellow-700",
      HIGH: "bg-orange-100 text-orange-700",
      CRITICAL: "bg-red-100 text-red-700",
    };
    return map[level] || "bg-gray-100 text-gray-700";
  }

  return (
    <>
      <Navbar />

      <div className="relative min-h-screen overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 z-[-1] bg-white bg-[radial-gradient(100%_60%_at_50%_0%,rgba(0,163,255,0.15)_0,rgba(0,163,255,0)_60%,rgba(0,163,255,0)_100%)]" />

        <div className="max-w-4xl mx-auto px-6 py-14">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-10"
          >
            <h2 className="text-4xl font-extrabold">
              Lung Disease Prediction
            </h2>
            <p className="mt-3 text-gray-600 text-lg">
              Upload a chest X-ray to analyze disease severity
            </p>
          </motion.div>

          {/* Upload Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl p-8"
          >
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Upload Box */}
              <label className="group cursor-pointer block border-2 border-dashed rounded-xl p-8 text-center hover:border-blue-500 transition">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <UploadCloud className="mx-auto text-blue-500" size={48} />
                <p className="mt-3 text-gray-600">
                  Click to upload chest X-ray
                </p>
                <p className="text-sm text-gray-400">
                  PNG / JPG supported
                </p>
              </label>

              {/* Preview */}
              {previewUrl && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-center"
                >
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="max-h-64 rounded-xl border shadow"
                  />
                </motion.div>
              )}

              {/* Submit */}
              <button
                type="submit"
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" />
                    Analyzing X-ray...
                  </>
                ) : (
                  <>
                    <ImageIcon />
                    Predict Disease
                  </>
                )}
              </button>
            </form>

            {/* Error */}
            {errorMessage && (
              <p className="mt-4 text-center text-red-600">
                {errorMessage}
              </p>
            )}
          </motion.div>

          {/* Result */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-10 bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl p-8"
            >
              <h3 className="text-2xl font-semibold mb-6">
                Prediction Result
              </h3>

              <div className="grid gap-4 text-lg">
                <p><strong>Organ:</strong> {result.organ}</p>
                <p><strong>Disease:</strong> {result.disease}</p>

                <p>
                  <strong>Criticality:</strong>{" "}
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${getBadge(
                      result.criticality
                    )}`}
                  >
                    {result.criticality}
                  </span>
                </p>

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

export default Lung;
