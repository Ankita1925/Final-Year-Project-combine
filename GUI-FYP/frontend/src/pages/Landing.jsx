////////////////////////////////////////////////////////////////////
//
// File Name : Landing.jsx
// Description : Public landing page for Disease Prediction System
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { Link } from "react-router-dom";

function Landing() {
  return (
    <div className="relative min-h-screen overflow-x-hidden text-gray-800">
      {/* ================= GLOBAL BACKGROUND ================= */}
      <div className="absolute top-0 z-[-2] h-screen w-screen bg-white bg-[radial-gradient(100%_50%_at_50%_0%,rgba(0,163,255,0.18)_0,rgba(0,163,255,0)_60%,rgba(0,163,255,0)_100%)]"></div>

      {/* ================= HEADER ================= */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-white/70 border-b">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-blue-600 tracking-wide">
            DiseaseAI
          </h1>

          <div className="flex gap-4">
            <Link
              to="/login"
              className="text-gray-700 font-medium hover:text-blue-600 transition"
            >
              Login
            </Link>
            <Link
              to="/login"
              className="bg-blue-600 text-white px-5 py-2 rounded-lg shadow hover:bg-blue-700 hover:scale-105 transition"
            >
              Get Started
            </Link>
          </div>
        </div>
      </header>

      {/* ================= HERO ================= */}
      <section className="max-w-7xl mx-auto px-6 py-24 grid md:grid-cols-2 gap-14 items-center">
        <div className="animate-fade-in">
          <h2 className="text-5xl font-extrabold leading-tight">
            AI-Powered <span className="text-blue-600">Disease Risk</span>
            <br />
            Prediction Platform
          </h2>

          <p className="mt-6 text-lg text-gray-600">
            DiseaseAI is an intelligent medical decision-support platform
            designed to assist clinicians in early detection, severity
            assessment, and treatment planning for critical diseases.
          </p>

          <div className="mt-10 flex gap-4">
            <Link
              to="/login"
              className="bg-blue-600 text-white px-8 py-3 rounded-xl shadow-lg hover:bg-blue-700 hover:scale-105 transition"
            >
              Access Platform
            </Link>

            <a
              href="#diseases"
              className="border border-blue-600 text-blue-600 px-8 py-3 rounded-xl hover:bg-blue-50 transition"
            >
              Explore Diseases
            </a>
          </div>
        </div>

        <div className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl shadow-xl animate-slide-up">
          <h3 className="text-xl font-semibold mb-4">
            Supported Disease Modules
          </h3>

          <ul className="space-y-3 text-gray-700 text-lg">
            <li>🫁 Lung Disease Severity Prediction</li>
            <li>🧬 Kidney Failure Stage Classification</li>
            <li>🫀 Heart Risk Assessment</li>
            <li>🩺 Liver Criticality Analysis</li>
            <li>📊 Treatment & Transplant Recommendation</li>
          </ul>
        </div>
      </section>

      {/* ================= DISEASE INFO ================= */}
      <section
        id="diseases"
        className="bg-white/70 backdrop-blur-xl py-20"
      >
        <div className="max-w-7xl mx-auto px-6">
          <h3 className="text-4xl font-bold text-center mb-14">
            Diseases We Focus On
          </h3>

          <div className="grid md:grid-cols-4 gap-8">
            {[
              {
                title: "Lung Diseases",
                desc: "Detection of pneumonia, tuberculosis, and viral infections using imaging and AI models.",
              },
              {
                title: "Kidney Disorders",
                desc: "Stage-wise prediction of chronic kidney disease to assist early intervention.",
              },
              {
                title: "Heart Conditions",
                desc: "Risk stratification for cardiovascular diseases using clinical parameters.",
              },
              {
                title: "Liver Failure",
                desc: "Assessment of liver damage severity and transplant necessity.",
              },
            ].map((item, index) => (
              <div
                key={index}
                className="p-6 rounded-xl bg-white shadow hover:shadow-xl hover:-translate-y-1 transition"
              >
                <h4 className="font-semibold text-lg mb-2">
                  {item.title}
                </h4>
                <p className="text-gray-600 text-sm">
                  {item.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================= TEAM ================= */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-6">
          <h3 className="text-4xl font-bold text-center mb-12">
            Project Team
          </h3>

          <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-8">
            {[
              "Pradhumnya Changdev Kalsait",
              "Ankita Sawant",
              "Onkar Bansode",
              "Ayush Mahadik",
            ].map((name, index) => (
              <div
                key={index}
                className="bg-white/80 backdrop-blur-xl p-6 rounded-xl shadow hover:shadow-xl hover:-translate-y-1 transition text-center"
              >
                <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-blue-100 flex items-center justify-center text-2xl font-bold text-blue-600">
                  {name.charAt(0)}
                </div>
                <h4 className="font-semibold">{name}</h4>
                <p className="text-sm text-gray-500">
                  Computer Engineering
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================= FOOTER ================= */}
      <footer className="bg-gray-900 text-gray-300 py-10">
        <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-8">
          <div>
            <h4 className="text-xl font-bold text-white mb-3">
              DiseaseAI
            </h4>
            <p className="text-sm">
              An AI-driven disease prediction and clinical
              decision support system developed as a
              Final Year Engineering Project.
            </p>
          </div>

          <div>
            <h5 className="font-semibold text-white mb-2">
              Quick Links
            </h5>
            <ul className="space-y-1 text-sm">
              <li>
                <Link to="/login" className="hover:text-white">
                  Login
                </Link>
              </li>
              <li>
                <a href="#diseases" className="hover:text-white">
                  Diseases
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h5 className="font-semibold text-white mb-2">
              Project Info
            </h5>
            <p className="text-sm">
              Guided by academic mentors and built using
              Machine Learning & Web Technologies.
            </p>
          </div>
        </div>

        <div className="mt-8 text-center text-sm text-gray-400">
          © 2026 DiseaseAI — Developed by Pradhumnya Changdev Kalsait & Team
        </div>
      </footer>
    </div>
  );
}

export default Landing;
