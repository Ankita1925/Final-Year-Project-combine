import React, { useState } from 'react';

const Heart = () => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    height: '',
    weight: '',
    systolic: '',
    diastolic: '',
    cholesterol: '',
    glucose: '',
    smoking: '',
    alcohol: '',
    active: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const fillSampleData = () => {
    setFormData({
      age: '55',
      gender: '2',
      height: '175',
      weight: '80',
      systolic: '140',
      diastolic: '90',
      cholesterol: '2',
      glucose: '1',
      smoking: '0',
      alcohol: '0',
      active: '1'
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/heart/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          Age: parseFloat(formData.age),
          Gender: parseInt(formData.gender),
          "Height(cm)": parseFloat(formData.height),
          "Weight(kg)": parseFloat(formData.weight),
          Systolic_Blood_Pressure: parseFloat(formData.systolic),
          Diastolic_Blood_Pressure: parseFloat(formData.diastolic),
          Cholesterol: parseInt(formData.cholesterol),
          Glucose: parseInt(formData.glucose),
          Smoking: parseInt(formData.smoking),
          Alcohol: parseInt(formData.alcohol),
          Physical_Activity: parseInt(formData.active)
        }),
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.result);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    const colors = {
      'Low Risk': '#27ae60',
      'Moderate Risk': '#f39c12',
      'High Risk': '#e67e22',
      'Critical Risk': '#e74c3c'
    };
    return colors[level] || '#666';
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <div className="text-center mb-8 p-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white">
        <h1 className="text-3xl font-bold mb-2">🫀 Heart Disease Prediction</h1>
        <p className="text-lg opacity-90">AI-Powered Medical Diagnosis System</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Personal Information */}
          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Age (years)</label>
            <input
              type="number"
              name="age"
              value={formData.age}
              onChange={handleChange}
              required
              min="20"
              max="100"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g. 50"
            />
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Gender</label>
            <select
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              required
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Select Gender</option>
              <option value="1">Female</option>
              <option value="2">Male</option>
            </select>
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Height (cm)</label>
            <input
              type="number"
              name="height"
              value={formData.height}
              onChange={handleChange}
              required
              min="100"
              max="250"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g. 175"
            />
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Weight (kg)</label>
            <input
              type="number"
              name="weight"
              value={formData.weight}
              onChange={handleChange}
              required
              min="30"
              max="200"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g. 75"
            />
          </div>

          {/* Clinical Measurements */}
          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Systolic BP (mmHg)</label>
            <input
              type="number"
              name="systolic"
              value={formData.systolic}
              onChange={handleChange}
              required
              min="90"
              max="200"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g. 120"
            />
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Diastolic BP (mmHg)</label>
            <input
              type="number"
              name="diastolic"
              value={formData.diastolic}
              onChange={handleChange}
              required
              min="60"
              max="130"
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g. 80"
            />
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Cholesterol Level</label>
            <select
              name="cholesterol"
              value={formData.cholesterol}
              onChange={handleChange}
              required
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Select Level</option>
              <option value="1">Normal</option>
              <option value="2">Above Normal</option>
              <option value="3">Well Above Normal</option>
            </select>
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Glucose Level</label>
            <select
              name="glucose"
              value={formData.glucose}
              onChange={handleChange}
              required
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Select Level</option>
              <option value="1">Normal</option>
              <option value="2">Above Normal</option>
              <option value="3">Well Above Normal</option>
            </select>
          </div>

          {/* Lifestyle Factors */}
          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Smoking Status</label>
            <select
              name="smoking"
              value={formData.smoking}
              onChange={handleChange}
              required
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Select Status</option>
              <option value="0">Non-smoker</option>
              <option value="1">Smoker</option>
            </select>
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Alcohol Intake</label>
            <select
              name="alcohol"
              value={formData.alcohol}
              onChange={handleChange}
              required
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Select Intake</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

          <div className="form-group">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Physical Activity</label>
            <select
              name="active"
              value={formData.active}
              onChange={handleChange}
              required
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Select Activity Level</option>
              <option value="0">Inactive</option>
              <option value="1">Active</option>
            </select>
          </div>
        </div>

        <div className="flex justify-center gap-4 mt-8">
          <button
            type="button"
            onClick={fillSampleData}
            className="px-6 py-3 border-2 border-blue-500 text-blue-500 rounded-full hover:bg-blue-50 transition-colors"
          >
            📋 Fill Sample Data
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold rounded-full hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50"
          >
            {loading ? 'Analyzing...' : 'Predict Heart Disease Type'}
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-8 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="mt-8 p-6 bg-gray-50 rounded-xl border-l-4 border-blue-500 animate-fade-in">
          <h3 className="text-2xl font-bold text-gray-800 mb-4">Analysis Result</h3>
          
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm">
                Status: <strong>{result.presence}</strong>
              </span>
              <span 
                className="px-3 py-1 text-white rounded-full text-sm"
                style={{ backgroundColor: getRiskColor(result.risk_level) }}
              >
                {result.risk_level}
              </span>
            </div>

            <div className="text-2xl font-bold mb-2 text-gray-800">
              {result.disease_type}
            </div>
            
            <div className="text-gray-600 mb-6">
              Confidence: {result.confidence}%
            </div>

            <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-600 mb-6">
              <h4 className="text-green-800 font-bold mb-2">🏥 Medical Decision Support</h4>
              <p className="mb-2"><strong>Recommended Action:</strong> {result.medical_action}</p>
              <p className="text-gray-600 text-sm">{result.recommendation}</p>
            </div>

            {result.probabilities && (
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(result.probabilities).map(([key, value]) => (
                  <div key={key} className="flex justify-between p-2 bg-gray-100 rounded">
                    <span>{key}</span>
                    <span className="font-bold">{value}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Heart;
