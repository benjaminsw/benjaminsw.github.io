import { ProfileData } from './types';

// =====================================================================
// MAIN CONFIGURATION FILE
// =====================================================================
// This is the active source of truth for your profile data.
// Edit the values below to update your website.
// =====================================================================

export const profileData: ProfileData = {
  "name": "Benjamin Wiriyapong",
  //"photoUrl": "https://picsum.photos/400/400",
  "photoUrl": "https://picsum.photos/400/400",
  "currentRole": "PhD Student",
  "email": "benjaminsw@live.com",
  "website": "www.bensw.xyz",
  "socials": {
    "linkedin": "https://linkedin.com",
    "github": "https://github.com",
    "scholar": "https://scholar.google.com",
    "twitter": "https://twitter.com"
  },
  "institution": {
    "name": "Cardiff University",
    "department": "Department of Computer Science",
    "location": "Cardiff, Wales",
    "website": "https://mit.edu"
  },
  "advisor": {
    "name": "Dr Oktay Karakus",
    "title": "Assistant Professor",
    "profileUrl": "#"
  },
  "researchFocus": {
    "area": "Variational Inference",
    "keywords": [
      "Bayesian Inference",
      "Variational Inference",
      "Mixture Model",
      "Normalising Flows",
      "Inverse Problem"
    ]
  },
  "thesisTitle": "Beyond Bayesian Machine Learning: Self-Learning Networks From Imaging Uncertainties",
  "expectedGraduation": "Dec 2026",
  "bio": "I am exploring how generative models can augment rather than replace human creativity. My work focuses on designing interaction paradigms that allow users to steer AI outputs with semantic precision."
};
