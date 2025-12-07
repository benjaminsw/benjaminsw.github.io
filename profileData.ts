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
    "department": "School of Computer Science and Informatics",
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
      "Mixture Model",
      "Uncertainty Quantification",
      "Normalising Flows",
      "Inverse Problem"
    ]
  },
  "thesisTitle": "Beyond Bayesian Machine Learning: Self-Learning Networks From Imaging Uncertainties",
  "expectedGraduation": "Dec 2026",
  "bio": "I develop probabilistic machine learning methods that make inference more stable, interpretable, and geometry aware. My work on Adaptive Mixture Flow Variational Inference introduces a two-stage framework that combines heterogeneous normalising flows with a Simplex EMA weighting scheme for robust multimodal posterior approximation. I extend these ideas to imaging through the Conditional Sequential Mixture of Flows, which adds data-dependent gating, measurement-consistency layers, and hybrid training objectives for tasks like super-resolution and SAR despeckling. My latest direction, BO-CSMF, jointly infers both images and acquisition parameters, enabling physics-adaptive inference under operator uncertainty."
};
