<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1Hva_sMnZtfQPBaZdYkevhWY0ryFuekoq

# Academic Profile Website

A minimalist, single-page academic portfolio showcasing current PhD research and position.

## Features

- Clean, professional design with animated backgrounds
- Fully responsive layout
- Single-source configuration via `profileData.ts`
- Automatic deployment to GitHub Pages
- Built with React, TypeScript, and Tailwind CSS

## Local Development

**Prerequisites:** Node.js 18+
```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

Visit `http://localhost:3000`

## Customisation

Edit `profileData.ts` to update all profile information:
```typescript
export const profileData: ProfileData = {
  name: "Your Name",
  photoUrl: "url-to-photo",
  currentRole: "PhD Student",
  email: "your@email.com",
  // ... etc
};
```

All changes automatically reflect on the site.

## Deployment

Pushes to `master` branch automatically deploy to GitHub Pages via GitHub Actions.

**Set up:**
1. Enable GitHub Pages in repository settings
2. Set source to "GitHub Actions"
3. Push to `master` branch

Site will be live at: `https://yourusername.github.io/repo-name`

## Build
```bash
npm run build
```

Output in `./dist`

## Tech Stack

- React 19
- TypeScript 5.8
- Vite 6.4
- Tailwind CSS (via CDN)
- GitHub Pages

## Structure
```
├── App.tsx                    # Main component
├── profileData.ts             # Configuration (edit this!)
├── components/
│   ├── CurrentPositionCard.tsx
│   ├── Layout.tsx
│   └── Icons.tsx
└── types.ts                   # TypeScript interfaces
```

## License

MIT

https://benjaminsw.github.io

