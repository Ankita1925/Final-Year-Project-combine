////////////////////////////////////////////////////////////////////
//
// File Name : Dashboard.jsx
// Description : Doctor dashboard with navbar and organ selection
// Author : Pradhumnya Changdev Kalsait
// Date : 17/01/26
//
////////////////////////////////////////////////////////////////////

import Navbar from "../components/Navbar";
import { useNavigate } from "react-router-dom";

function Dashboard() {
  const navigate = useNavigate();

  function handleOrganSelection(organName) {
    navigate(`/organ/${organName}`);
  }

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 p-10">
        <h2 className="text-3xl font-bold mb-8">
          Select Organ for Disease Prediction
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {["lung", "liver", "kidney", "heart"].map((organ) => (
            <div
              key={organ}
              onClick={() => handleOrganSelection(organ)}
              className="cursor-pointer bg-white p-6 rounded-lg shadow hover:shadow-lg transition"
            >
              <h3 className="text-xl font-semibold capitalize">
                {organ}
              </h3>
              <p className="text-gray-600 mt-2">
                Disease prediction
              </p>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

export default Dashboard;
