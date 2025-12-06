import React, { useState } from 'react';
import Layout from './components/Layout';
import CurrentPositionCard from './components/CurrentPositionCard';
import { PROFILE } from './constants';
import { MailIcon, GithubIcon, LinkedinIcon, BookIcon, TwitterIcon, LinkIcon } from './components/Icons';

const App: React.FC = () => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <Layout>
      {/* Header / Intro */}
      <div className="flex flex-col items-center mb-12 text-center">
        <div 
          className="relative mb-6 group"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <div className={`absolute -inset-0.5 bg-gradient-to-r from-academic-400 to-indigo-500 rounded-full blur opacity-50 group-hover:opacity-75 transition duration-500 ${isHovered ? 'scale-105' : 'scale-100'}`}></div>
          <img 
            src={PROFILE.photoUrl} 
            alt={PROFILE.name} 
            className="relative w-32 h-32 sm:w-40 sm:h-40 rounded-full object-cover border-4 border-white shadow-lg"
          />
        </div>
        
        <h1 className="text-4xl sm:text-5xl font-bold text-slate-900 tracking-tight mb-2 font-serif">
          {PROFILE.name}
        </h1>
        <p className="text-lg text-slate-500 max-w-2xl mx-auto">
          Researcher • Scholar • Technologist
        </p>

        {/* Social / Contact Links */}
        <div className="flex items-center gap-4 mt-6">
            {PROFILE.socials.scholar && (
              <div className="p-2 text-slate-400 cursor-default" aria-label="Google Scholar">
                <BookIcon className="w-6 h-6" />
              </div>
            )}
            {PROFILE.socials.github && (
              <div className="p-2 text-slate-400 cursor-default" aria-label="GitHub">
                <GithubIcon className="w-6 h-6" />
              </div>
            )}
             {PROFILE.socials.twitter && (
              <div className="p-2 text-slate-400 cursor-default" aria-label="Twitter">
                <TwitterIcon className="w-6 h-6" />
              </div>
            )}
            {PROFILE.socials.linkedin && (
              <div className="p-2 text-slate-400 cursor-default" aria-label="LinkedIn">
                <LinkedinIcon className="w-6 h-6" />
              </div>
            )}
            {PROFILE.website && (
              <div className="p-2 text-slate-400 cursor-default" aria-label="Personal Website">
                 <LinkIcon className="w-6 h-6" />
              </div>
            )}
        </div>
      </div>

      {/* Main Content: The Single Current Position */}
      <main className="max-w-4xl mx-auto animate-fade-in-up">
        <CurrentPositionCard 
          currentRole={PROFILE.currentRole}
          institution={PROFILE.institution}
          advisor={PROFILE.advisor}
          researchFocus={PROFILE.researchFocus}
          thesisTitle={PROFILE.thesisTitle}
          expectedGraduation={PROFILE.expectedGraduation}
          bio={PROFILE.bio}
        />

        {/* Action Button - e.g. Contact or Download CV */}
        <div className="mt-12 flex justify-center">
            <a 
              href={`mailto:${PROFILE.email}`}
              className="group relative inline-flex items-center justify-center px-8 py-3 text-base font-medium text-white transition-all duration-200 bg-slate-900 rounded-full hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-900"
            >
              <MailIcon className="w-5 h-5 mr-2 group-hover:-translate-y-0.5 transition-transform" />
              Get in Touch
            </a>
        </div>
      </main>
      
      <footer className="mt-20 text-center text-slate-400 text-sm pb-8">
        <p>© {new Date().getFullYear()} {PROFILE.name}. All rights reserved.</p>
      </footer>
    </Layout>
  );
};

export default App;