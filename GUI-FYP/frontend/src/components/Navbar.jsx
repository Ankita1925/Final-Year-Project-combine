////////////////////////////////////////////////////////////////////
//
// File Name : Navbar.jsx
// Description : Common navigation bar for all portals
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { useContext } from "react";
import { useNavigate, Link } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";

/**
 * ////////////////////////////////////////////////////////////////
 *
 * Function Name : Navbar
 * Description   : Displays application-wide navigation bar
 * Author        : Pradhumnya Changdev Kalsait
 * Date          : 17/01/26
 *
 * ////////////////////////////////////////////////////////////////
 */
function Navbar() {
  const { userRole, logout } = useContext(AuthContext);
  const navigate = useNavigate();
  

  function handleLogout() {
    logout();
    navigate("/");
  }

  return (
    <nav className="bg-cardbg shadow-card px-6 py-4 flex justify-between items-center">
      {/* ================= LEFT ================= */}
      <div className="flex items-center gap-6">
        <Link
          to="/dashboard"
          className="text-primary font-bold text-xl"
        >
          DiseaseAI
        </Link>

        {/* {userRole === "DOCTOR" && (

        )} */}

        {userRole === "ADMIN" && (
          <Link
            to="/admin"
            className="text-textsecondary hover:text-primary transition"
          >
            Admin Panel
          </Link>
        )}
      </div>

      {/* ================= RIGHT ================= */}
      <div className="flex items-center gap-4">
        {userRole && (
          <span className="text-sm bg-primary-light text-primary px-3 py-1 rounded-full">
            {userRole}
          </span>
        )}

        <button
          onClick={handleLogout}
          className="btn-primary"
        >
          Logout
        </button>
      </div>
    </nav>
  );
}

export default Navbar;
