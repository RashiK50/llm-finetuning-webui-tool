/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0F111A',
        surface: '#1E212B',
        surfaceHover: '#2A2E39',
        primary: '#3B82F6',
        primaryHover: '#2563EB',
        accent: '#10B981',
        textMain: '#F8FAFC',
        textMuted: '#94A3B8',
        border: '#334155',
        error: '#EF4444',
        success: '#22C55E',
        warning: '#F59E0B'
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
