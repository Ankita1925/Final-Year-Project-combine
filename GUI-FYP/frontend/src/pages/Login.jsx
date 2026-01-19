////////////////////////////////////////////////////////////////////
//
// File Name : Login.jsx
// Description : Authentication page with Sign In / Sign Up (Enhanced UI)
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosInstance";
import { AuthContext } from "../context/AuthContext";

function Login() {
  const [isSignIn, setIsSignIn] = useState(true);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const { login } = useContext(AuthContext);
  const navigate = useNavigate();

  async function handleSubmit(event) {
    event.preventDefault();
    setErrorMessage("");
    setLoading(true);

    try {
      if (!isSignIn && password !== confirmPassword) {
        throw new Error("Passwords do not match");
      }

      const endpoint = isSignIn ? "/auth/login" : "/auth/register";

      const response = await axiosInstance.post(endpoint, {
        email,
        password,
      });

      if (isSignIn) {
        const token =
          response.data.access_token ||
          response.data.token ||
          response.data.accessToken;

        if (!token) {
          throw new Error("JWT token missing");
        }

        login(token);
        navigate("/dashboard");
      } else {
        setIsSignIn(true);
      }
    } catch (error) {
      setErrorMessage(
        error.message || "Authentication failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-100 via-indigo-100 to-purple-100">
      <div className="relative w-full max-w-md">
        {/* Glass Card */}
        <div className="backdrop-blur-xl bg-white/80 rounded-2xl shadow-2xl border border-white/30 overflow-hidden animate-fade-in">
          
          {/* ================= TABS ================= */}
          <div className="flex relative">
            <div
              className={`absolute top-0 h-full w-1/2 bg-primary transition-transform duration-300 ${
                isSignIn ? "translate-x-0" : "translate-x-full"
              }`}
            />

            <button
              onClick={() => setIsSignIn(true)}
              className={`relative w-1/2 py-4 font-semibold z-10 transition ${
                isSignIn ? "text-white" : "text-primary"
              }`}
            >
              Sign In
            </button>

            <button
              onClick={() => setIsSignIn(false)}
              className={`relative w-1/2 py-4 font-semibold z-10 transition ${
                !isSignIn ? "text-white" : "text-primary"
              }`}
            >
              Sign Up
            </button>
          </div>

          {/* ================= FORM ================= */}
          <form
            onSubmit={handleSubmit}
            className="p-8 space-y-5"
          >
            <h2 className="text-3xl font-bold text-center text-gray-800">
              {isSignIn ? "Welcome Back 👋" : "Create Account 🚀"}
            </h2>

            <p className="text-center text-gray-500 text-sm">
              {isSignIn
                ? "Sign in to access DiseaseAI Dashboard"
                : "Register to start using DiseaseAI"}
            </p>

            {errorMessage && (
              <div className="bg-red-100 text-red-700 px-4 py-2 rounded text-sm text-center animate-shake">
                {errorMessage}
              </div>
            )}

            {/* Email */}
            <div className="group">
              <label className="block text-sm font-medium text-gray-600 mb-1">
                Email
              </label>
              <input
                type="email"
                className="w-full px-4 py-3 rounded-lg border bg-white/70 focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition shadow-sm"
                placeholder="doctor@test.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            {/* Password */}
            <div className="group">
              <label className="block text-sm font-medium text-gray-600 mb-1">
                Password
              </label>
              <input
                type="password"
                className="w-full px-4 py-3 rounded-lg border bg-white/70 focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition shadow-sm"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            {/* Confirm Password */}
            {!isSignIn && (
              <div className="group animate-slide-down">
                <label className="block text-sm font-medium text-gray-600 mb-1">
                  Confirm Password
                </label>
                <input
                  type="password"
                  className="w-full px-4 py-3 rounded-lg border bg-white/70 focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition shadow-sm"
                  placeholder="••••••••"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>
            )}

            {/* Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-lg bg-primary text-white font-semibold tracking-wide shadow-lg hover:shadow-xl hover:scale-[1.02] transition-all duration-300 disabled:opacity-70"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </span>
              ) : isSignIn ? (
                "Sign In"
              ) : (
                "Create Account"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Login;
