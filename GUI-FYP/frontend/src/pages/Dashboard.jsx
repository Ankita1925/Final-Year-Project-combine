////////////////////////////////////////////////////////////////////
 //
 // File Name : Dashboard.jsx
 // Description : Doctor dashboard with animated organ selection
 // Author      : Pradhumnya Changdev Kalsait
 // Date        : 17/01/26
 //
 ////////////////////////////////////////////////////////////////////

import Navbar from "../components/Navbar";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity,
  HeartPulse,
  Droplets,
  Stethoscope,
  Users,
  BarChart3,
  ShieldCheck,
  AlertTriangle,
} from "lucide-react";

function Dashboard() {
  const navigate = useNavigate();

  function handleOrganSelection(organName) {
    navigate(`/organ/${organName}`);
  }

  /* ================= ORGAN CONFIG ================= */
  const organs = [
    {
      key: "lung",
      title: "Lung",
      desc: "Respiratory disease severity analysis",
      icon: <Activity size={36} />,
      color: "from-sky-400 to-blue-600",
    },
    {
      key: "heart",
      title: "Heart",
      desc: "Cardiovascular risk prediction",
      icon: <HeartPulse size={36} />,
      color: "from-rose-400 to-red-600",
    },
    {
      key: "kidney",
      title: "Kidney",
      desc: "CKD stage & failure prediction",
      icon: <Droplets size={36} />,
      color: "from-emerald-400 to-green-600",
    },
    {
      key: "liver",
      title: "Liver",
      desc: "Liver damage & transplant risk",
      icon: <Stethoscope size={36} />,
      color: "from-amber-400 to-orange-600",
    },
  ];

  /* ================= STATS ================= */
  const stats = [
    {
      label: "Patients Today",
      value: "42",
      icon: <Users />,
      color: "from-blue-500 to-sky-500",
    },
    {
      label: "Predictions Done",
      value: "168",
      icon: <BarChart3 />,
      color: "from-green-500 to-emerald-500",
    },
    {
      label: "Model Accuracy",
      value: "94.6%",
      icon: <ShieldCheck />,
      color: "from-purple-500 to-indigo-500",
    },
    {
      label: "Critical Alerts",
      value: "7",
      icon: <AlertTriangle />,
      color: "from-red-500 to-rose-500",
    },
  ];

  return (
    <>
      <Navbar />

      <div className="relative min-h-screen overflow-hidden">
        {/* ================= BACKGROUND ================= */}
        <div className="absolute inset-0 z-[-1] bg-white bg-[radial-gradient(100%_60%_at_50%_0%,rgba(0,163,255,0.15)_0,rgba(0,163,255,0)_60%,rgba(0,163,255,0)_100%)]" />

        <div className="max-w-7xl mx-auto px-6 py-14">
          {/* ================= HEADER ================= */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-12 text-center"
          >
            <h2 className="text-4xl font-extrabold tracking-tight">
              Doctor Dashboard
            </h2>
            <p className="mt-3 text-gray-600 text-lg">
              AI-powered clinical decision support overview
            </p>
          </motion.div>

          {/* ================= STATS PANEL ================= */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-14"
          >
            {stats.map((stat, idx) => (
              <div
                key={idx}
                className="relative bg-white/80 backdrop-blur-xl rounded-2xl p-5 shadow-lg overflow-hidden"
              >
                <div
                  className={`absolute inset-0 opacity-20 bg-gradient-to-br ${stat.color}`}
                />
                <div className="relative flex items-center gap-4">
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center text-white bg-gradient-to-br ${stat.color}`}
                  >
                    {stat.icon}
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">
                      {stat.label}
                    </p>
                    <p className="text-2xl font-bold">
                      {stat.value}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </motion.div>

          {/* ================= ORGAN CARDS ================= */}
          <motion.div
            initial="hidden"
            animate="show"
            variants={{
              hidden: { opacity: 0 },
              show: {
                opacity: 1,
                transition: { staggerChildren: 0.15 },
              },
            }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8"
          >
            {organs.map((organ) => (
              <motion.div
                key={organ.key}
                variants={{
                  hidden: { opacity: 0, y: 40 },
                  show: { opacity: 1, y: 0 },
                }}
                whileHover={{ scale: 1.05 }}
                onClick={() => handleOrganSelection(organ.key)}
                className="group cursor-pointer relative bg-white/80 backdrop-blur-xl rounded-2xl p-6 shadow-xl transition-all"
              >
                {/* Particle Glow */}
                <div
                  className={`absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-30 blur-2xl transition bg-gradient-to-br ${organ.color}`}
                />

                {/* Icon */}
                <div
                  className={`relative w-16 h-16 rounded-xl flex items-center justify-center text-white bg-gradient-to-br ${organ.color} shadow-md mb-5`}
                >
                  {organ.icon}
                </div>

                <h3 className="text-xl font-semibold mb-1">
                  {organ.title}
                </h3>
                <p className="text-gray-600 text-sm">
                  {organ.desc}
                </p>

                <div className="mt-6 text-sm font-medium text-blue-600 opacity-0 group-hover:opacity-100 transition">
                  Start Prediction →
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* ================= FOOTER ================= */}
        <div className="absolute bottom-4 w-full text-center text-sm text-gray-400">
          DiseaseAI © 2026 — AI-Driven Clinical Decision Support
        </div>
      </div>
    </>
  );
}

export default Dashboard;
