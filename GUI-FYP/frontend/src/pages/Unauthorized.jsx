////////////////////////////////////////////////////////////////////
//
// File Name : Unauthorized.jsx
// Description : Unauthorized access page
// Author      : Pradhumnya Changdev Kalsait
// Date        : 17/01/26
//
////////////////////////////////////////////////////////////////////

import { Link } from "react-router-dom";

/**
 * ////////////////////////////////////////////////////////////////
 *
 * Function Name : Unauthorized
 * Description   : Informs user of insufficient permissions
 * Author        : Pradhumnya Changdev Kalsait
 * Date          : 17/01/26
 *
 * ////////////////////////////////////////////////////////////////
 */
function Unauthorized() {

  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-gray-100">
      <h1 className="text-3xl font-bold text-red-600">
        Access Denied
      </h1>

      <p className="mt-4 text-gray-600">
        You do not have permission to view this page.
      </p>

      <Link
        to="/dashboard"
        className="mt-6 text-blue-600 underline"
      >
        Go back to dashboard
      </Link>
    </div>
  );
}

export default Unauthorized;
