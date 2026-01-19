////////////////////////////////////////////////////////////////////
//
// File Name : Login.jsx
// Description : Authentication page with Sign In / Sign Up
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosInstance";
import { AuthContext } from "../context/AuthContext";

/**
 * ////////////////////////////////////////////////////////////////
 *
 * Function Name : Login
 * Description   : Handles user authentication with Sign In / Sign Up
 * Author        : Pradhumnya Changdev Kalsait
 * Date          : 17/01/26
 *
 * ////////////////////////////////////////////////////////////////
 */
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
    <div className="min-h-screen flex items-center justify-center bg-appbg">
      <div className="bg-cardbg w-full max-w-md rounded-xl shadow-card overflow-hidden">
        {/* ================= HEADER ================= */}
        <div className="flex">
          <button
            onClick={() => setIsSignIn(true)}
            className={`w-1/2 py-4 text-center font-semibold transition ${
              isSignIn
                ? "bg-primary text-white"
                : "bg-primary-light text-primary"
            }`}
          >
            Sign In
          </button>

          <button
            onClick={() => setIsSignIn(false)}
            className={`w-1/2 py-4 text-center font-semibold transition ${
              !isSignIn
                ? "bg-primary text-white"
                : "bg-primary-light text-primary"
            }`}
          >
            Sign Up
          </button>
        </div>

        {/* ================= FORM ================= */}
        <form
          onSubmit={handleSubmit}
          className="p-8 space-y-4 animate-fade-in"
        >
          <h2 className="text-2xl font-bold text-center">
            {isSignIn ? "Welcome Back" : "Create an Account"}
          </h2>

          <p className="text-center text-textsecondary text-sm">
            {isSignIn
              ? "Sign in to continue to DiseaseAI"
              : "Register to access DiseaseAI platform"}
          </p>

          {errorMessage && (
            <p className="text-danger text-sm text-center">
              {errorMessage}
            </p>
          )}

          <div>
            <label className="block text-sm font-medium">
              Email
            </label>
            <input
              type="email"
              className="mt-1 w-full px-3 py-2 border rounded-md focus:outline-none focus:ring focus:border-primary"
              placeholder="doctor@test.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium">
              Password
            </label>
            <input
              type="password"
              className="mt-1 w-full px-3 py-2 border rounded-md focus:outline-none focus:ring focus:border-primary"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {!isSignIn && (
            <div className="transition-all duration-300">
              <label className="block text-sm font-medium">
                Confirm Password
              </label>
              <input
                type="password"
                className="mt-1 w-full px-3 py-2 border rounded-md focus:outline-none focus:ring focus:border-primary"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary py-2"
          >
            {loading
              ? "Please wait..."
              : isSignIn
              ? "Sign In"
              : "Sign Up"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default Login;
