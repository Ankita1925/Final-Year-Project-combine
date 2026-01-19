////////////////////////////////////////////////////////////////////
//
// File Name : OrganPlaceholder.jsx
// Description : Organ-specific placeholder page with navbar
// Author : Pradhumnya Changdev Kalsait
// Date : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { useParams } from "react-router-dom";
import Navbar from "../components/Navbar";

function OrganPlaceholder() {
  const { organName } = useParams();

  return (
    <>
      <Navbar />

      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <h2 className="text-2xl font-bold">
          {organName.toUpperCase()} Form Coming Soon
        </h2>
      </div>
    </>
  );
}

export default OrganPlaceholder;
