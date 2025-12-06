export interface Institution {
  name: string;
  department: string;
  location: string;
  logoUrl?: string;
  website?: string;
}

export interface Advisor {
  name: string;
  title: string;
  profileUrl?: string;
}

export interface ResearchFocus {
  area: string;
  keywords: string[];
}

export interface ProfileData {
  name: string;
  photoUrl: string;
  currentRole: string; // e.g., "PhD Candidate"
  email: string;
  website: string;
  socials: {
    linkedin?: string;
    twitter?: string;
    github?: string;
    scholar?: string;
  };
  institution: Institution;
  advisor: Advisor;
  researchFocus: ResearchFocus;
  thesisTitle?: string;
  expectedGraduation?: string;
  bio: string;
}