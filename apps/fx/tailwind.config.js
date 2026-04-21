/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // FX-Range-Master dark navy palette
        bg: {
          900: '#0b1120',
          800: '#0f172a',
          700: '#1a2540',
          600: '#1e293b',
        },
        accent: {
          DEFAULT: '#4f8eff',
          50:  '#eef4ff',
          100: '#d9e8ff',
          200: '#bcd3ff',
          300: '#91b6ff',
          400: '#6496ff',
          500: '#4f8eff',
          600: '#2f67f0',
          700: '#2050d6',
          800: '#1a40ac',
          900: '#1a3a8a',
        },
        gain:   '#22c55e',
        loss:   '#ef4444',
        warn:   '#f59e0b',
        cyan:   '#06b6d4',
        orange: '#f97316',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
