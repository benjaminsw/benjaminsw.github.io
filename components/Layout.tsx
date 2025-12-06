import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen relative overflow-hidden bg-slate-50 text-slate-900 selection:bg-academic-200 selection:text-academic-900">
      {/* Abstract Background Shapes */}
      <div className="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-white to-slate-50 z-0" />
      <div className="absolute -top-40 -right-40 w-96 h-96 bg-academic-100 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob" />
      <div className="absolute top-0 -left-40 w-96 h-96 bg-indigo-100 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000" />
      <div className="absolute -bottom-40 left-20 w-96 h-96 bg-pink-50 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-4000" />
      
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-20">
        {children}
      </div>
    </div>
  );
};

export default Layout;