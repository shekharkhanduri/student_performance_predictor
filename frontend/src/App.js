import React, { useState } from 'react';
import StudentDashboard from './components/StudentDashboard';
import FacultyDashboard from './components/FacultyDashboard';

export default function App() {
  const [activeTab, setActiveTab] = useState('student');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-blue-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                <span className="text-blue-600 font-bold text-sm">SP</span>
              </div>
              <span className="text-white font-bold text-lg">
                Student Performance Predictor
              </span>
            </div>
            <div className="flex space-x-1 bg-blue-700 rounded-lg p-1">
              <button
                onClick={() => setActiveTab('student')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'student'
                    ? 'bg-white text-blue-600'
                    : 'text-blue-100 hover:text-white'
                }`}
              >
                Student Dashboard
              </button>
              <button
                onClick={() => setActiveTab('faculty')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'faculty'
                    ? 'bg-white text-blue-600'
                    : 'text-blue-100 hover:text-white'
                }`}
              >
                Faculty Dashboard
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'student' ? <StudentDashboard /> : <FacultyDashboard />}
      </main>
    </div>
  );
}
