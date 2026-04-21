/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}', '../../packages/ui/src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Shared trading platform palette
        bg:      { 900: '#0b1120', 800: '#0f172a', 700: '#131a2e', 600: '#192038' },
        accent:  { DEFAULT: '#4f8eff', 400: '#7aaaff' },
        gain:    '#22c55e',
        loss:    '#ef4444',
        warn:    '#f59e0b',
        orange:  '#f97316',
        cyan:    '#06b6d4',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
