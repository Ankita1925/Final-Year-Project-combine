////////////////////////////////////////////////////////////////////
//
// File Name : AdminDashboard.jsx
// Description : Admin-only dashboard page
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import Navbar from "../components/Navbar";


/**
 * ////////////////////////////////////////////////////////////////
 *
 * Function Name : AdminDashboard
 * Description   : Displays administrative controls and statistics
 * Author        : Pradhumnya Changdev Kalsait
 * Date          : 17/01/26
 *
 * ////////////////////////////////////////////////////////////////
*/

function AdminDashboard() {

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 p-10">
        <h1 className="text-3xl font-bold mb-4">
          Admin Dashboard
        </h1>

        <p className="text-gray-600">
          Manage users, models, and system configuration.
        </p>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded shadow">
            <h3 className="font-semibold">User Management</h3>
            <p className="text-sm text-gray-500 mt-2">
              View and manage registered users
            </p>
          </div>

          <div className="bg-white p-6 rounded shadow">
            <h3 className="font-semibold">Model Status</h3>
            <p className="text-sm text-gray-500 mt-2">
              Monitor ML model health
            </p>
          </div>

          <div className="bg-white p-6 rounded shadow">
            <h3 className="font-semibold">System Logs</h3>
            <p className="text-sm text-gray-500 mt-2">
              Review audit and access logs
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

export default AdminDashboard;
