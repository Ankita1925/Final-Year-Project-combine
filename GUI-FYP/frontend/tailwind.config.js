////////////////////////////////////////////////////////////////////
//
// File Name : tailwind.config.js
// Description : Centralized Tailwind theme configuration
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        /* ================= BRAND COLORS ================= */
        primary: {
          DEFAULT: "#2563EB", // blue-600
          dark: "#1E40AF",    // blue-800
          light: "#DBEAFE",   // blue-100
        },

        secondary: {
          DEFAULT: "#4F46E5", // indigo-600
          dark: "#3730A3",
        },

        /* ================= STATUS COLORS ================= */
        success: "#16A34A",   // green-600
        warning: "#F59E0B",   // amber-500
        danger: "#DC2626",    // red-600
        info: "#0EA5E9",      // sky-500

        /* ================= BACKGROUND ================= */
        appbg: "#F9FAFB",     // gray-50
        cardbg: "#FFFFFF",

        /* ================= TEXT ================= */
        textprimary: "#1F2937", // gray-800
        textsecondary: "#6B7280", // gray-500
      },

      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },

      boxShadow: {
        card: "0 4px 12px rgba(0,0,0,0.05)",
      },

      borderRadius: {
        xl: "1rem",
      },
    },
  },
  plugins: [],
};
