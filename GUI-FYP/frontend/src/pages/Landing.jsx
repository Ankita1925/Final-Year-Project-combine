////////////////////////////////////////////////////////////////////
//
// File Name : Landing.jsx
// Description : Public landing page for Disease Prediction System
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { Link } from "react-router-dom";

/**
 * ////////////////////////////////////////////////////////////////
 *
 * Function Name : Landing
 * Description   : Displays landing page with system overview
 * Author        : Pradhumnya Changdev Kalsait
 * Date          : 17/01/26
 *
 * ////////////////////////////////////////////////////////////////
 */
function Landing() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* ================= HEADER ================= */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-blue-600">
            DiseaseAI
          </h1>

          <div className="space-x-4">
            <Link
              to="/login"
              className="text-gray-700 hover:text-blue-600"
            >
              Login
            </Link>
            <Link
              to="/login"
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
            >
              Get Started
            </Link>
          </div>
        </div>
      </header>

      {/* ================= HERO ================= */}
      <section className="max-w-7xl mx-auto px-6 py-20 grid md:grid-cols-2 gap-10 items-center">
        <div>
          <h2 className="text-4xl font-bold text-gray-800 leading-tight">
            AI-Driven Disease Classification &  
            <span className="text-blue-600"> Criticality Prediction</span>
          </h2>

          <p className="mt-6 text-gray-600 text-lg">
            An intelligent decision support system for predicting
            disease severity and transplant necessity for
            Lung, Liver, Kidney, and Heart patients.
          </p>

          <div className="mt-8 flex gap-4">
            <Link
              to="/login"
              className="bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 transition"
            >
              Login as Doctor
            </Link>

            <Link
              to="/login"
              className="border border-blue-600 text-blue-600 px-6 py-3 rounded-md hover:bg-blue-50 transition"
            >
              Admin Access
            </Link>
          </div>
        </div>

        <div className="bg-white p-8 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">
            Supported Modules
          </h3>

          <ul className="space-y-3 text-gray-700">
            <li>🫁 Lung Disease Analysis</li>
            <li>🫀 Heart Disease Prediction</li>
            <li>🩺 Liver Criticality Assessment</li>
            <li>🧬 Kidney Failure Evaluation</li>
            <li>📊 Treatment / Transplant Decision</li>
          </ul>
        </div>
      </section>

      {/* ================= FEATURES ================= */}
      <section className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h3 className="text-3xl font-bold text-center mb-12">
            Why Choose DiseaseAI?
          </h3>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="p-6 border rounded-lg text-center">
              <h4 className="font-semibold text-lg">
                AI-Powered Models
              </h4>
              <p className="mt-2 text-gray-600">
                Multiple ML models analyze patient data
                for accurate disease classification.
              </p>
            </div>

            <div className="p-6 border rounded-lg text-center">
              <h4 className="font-semibold text-lg">
                Role-Based Access
              </h4>
              <p className="mt-2 text-gray-600">
                Secure access for Doctors and Admins
                with JWT authentication.
              </p>
            </div>

            <div className="p-6 border rounded-lg text-center">
              <h4 className="font-semibold text-lg">
                Clinical Decision Support
              </h4>
              <p className="mt-2 text-gray-600">
                Helps clinicians decide treatment
                or transplant necessity.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ================= FOOTER ================= */}
      <footer className="bg-gray-100 py-6">
        <div className="text-center text-gray-500 text-sm">
          © 2026 DiseaseAI — Final Year Project  
          <br />
          Developed by Pradhumnya Changdev Kalsait
        </div>
      </footer>
    </div>
  );
}

export default Landing;
