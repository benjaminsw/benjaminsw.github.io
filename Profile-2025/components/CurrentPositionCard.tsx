import React from 'react';
import { Institution, Advisor, ResearchFocus } from '../types';
import { 
  BuildingIcon, 
  MapPinIcon, 
  GraduationCapIcon, 
  BookIcon,
} from './Icons';

interface CurrentPositionCardProps {
  currentRole: string;
  institution: Institution;
  advisor: Advisor;
  researchFocus: ResearchFocus;
  thesisTitle?: string;
  expectedGraduation?: string;
  bio: string;
}

const CurrentPositionCard: React.FC<CurrentPositionCardProps> = ({ 
  currentRole, 
  institution, 
  advisor,
  researchFocus,
  thesisTitle,
  expectedGraduation,
  bio
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden transition-all duration-300 hover:shadow-2xl hover:shadow-academic-100/50">
      <div className="h-2 w-full bg-gradient-to-r from-academic-500 to-indigo-600" />
      
      <div className="p-8 sm:p-10">
        <div className="flex items-center space-x-3 mb-6">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-academic-100 text-academic-800 tracking-wide uppercase">
            <span className="w-2 h-2 rounded-full bg-academic-500 mr-2 animate-pulse"></span>
            Current Position
          </span>
        </div>

        <div className="space-y-8">
          {/* Main Title Area */}
          <div>
            <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 font-serif mb-2">
              {currentRole}
            </h2>
            <div className="flex flex-col sm:flex-row sm:items-center text-lg text-slate-600 gap-2 sm:gap-4">
              <div className="flex items-center">
                <BuildingIcon className="w-5 h-5 mr-2 text-academic-600" />
                <span className="font-medium">{institution.department}</span>
              </div>
              <div className="hidden sm:block text-slate-300">•</div>
              <div className="flex items-center">
                <span className="font-medium text-slate-800">{institution.name}</span>
              </div>
            </div>
             <div className="flex items-center mt-2 text-slate-500 text-sm">
                <MapPinIcon className="w-4 h-4 mr-1.5" />
                {institution.location}
              </div>
          </div>

          <hr className="border-slate-100" />

          {/* Details Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Left Column: Academic Details */}
            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Advisor</h3>
                <div className="flex items-start">
                  <div className="bg-slate-50 p-2 rounded-lg mr-3">
                     <GraduationCapIcon className="w-5 h-5 text-slate-700" />
                  </div>
                  <div>
                    <p className="font-semibold text-slate-900">{advisor.name}</p>
                    <p className="text-sm text-slate-500">{advisor.title}</p>
                  </div>
                </div>
              </div>

              {expectedGraduation && (
                <div>
                   <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Timeline</h3>
                   <p className="text-slate-700 font-medium">Expected Graduation: <span className="text-slate-900">{expectedGraduation}</span></p>
                </div>
              )}
            </div>

            {/* Right Column: Research Focus */}
            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Primary Research Area</h3>
                <div className="flex items-start">
                   <div className="bg-slate-50 p-2 rounded-lg mr-3">
                     <BookIcon className="w-5 h-5 text-slate-700" />
                  </div>
                  <div>
                    <p className="font-semibold text-slate-900">{researchFocus.area}</p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {researchFocus.keywords.map((keyword, idx) => (
                        <span key={idx} className="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium bg-indigo-50 text-indigo-700">
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Thesis / Bio Section */}
          <div className="bg-slate-50 rounded-xl p-6 border border-slate-100">
            {thesisTitle && (
              <div className="mb-4">
                <h3 className="text-sm font-semibold text-slate-500 mb-1">Thesis Topic</h3>
                <p className="text-lg font-medium text-slate-900 bold">"{thesisTitle}"</p>
                
              </div>
            )}
            <div>
               <h3 className="text-sm font-semibold text-slate-500 mb-2">About the Role</h3>
               <p className="text-slate-600 leading-relaxed">
                 {bio}
               </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CurrentPositionCard;