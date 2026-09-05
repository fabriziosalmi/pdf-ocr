import { defineConfig } from 'vitepress'

// Project page on GitHub Pages: https://fabriziosalmi.github.io/pdf-ocr/
const SITE = 'https://fabriziosalmi.github.io'
const BASE = '/pdf-ocr/'
const ORIGIN = SITE + BASE.replace(/\/$/, '')

// Described once, emitted as JSON-LD on every page. SoftwareApplication is the
// right type: this is a self-hosted application, not a hosted service.
const structuredData = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'pdf-ocr',
  description:
    'Self-hosted Flask web app that turns scanned PDFs into DOCX, TXT, Markdown or HTML using OCR.',
  applicationCategory: 'DeveloperApplication',
  operatingSystem: 'Linux, macOS, Windows (via Docker)',
  url: ORIGIN + '/',
  codeRepository: 'https://github.com/fabriziosalmi/pdf-ocr',
  downloadUrl: 'https://github.com/fabriziosalmi/pdf-ocr/releases',
  installUrl: ORIGIN + '/guide/quickstart',
  license: 'https://opensource.org/licenses/MIT',
  programmingLanguage: 'Python',
  isAccessibleForFree: true,
  offers: { '@type': 'Offer', price: '0', priceCurrency: 'EUR' },
  author: {
    '@type': 'Person',
    name: 'Fabrizio Salmi',
    url: 'https://github.com/fabriziosalmi',
  },
}

export default defineConfig({
  title: 'pdf-ocr',
  description:
    'Self-hosted Flask web app that turns scanned PDFs into DOCX, TXT, Markdown or HTML using OCR.',
  lang: 'en-GB',
  base: BASE,
  cleanUrls: true,
  lastUpdated: true,

  // Fail the build on a broken internal link rather than shipping one. The one
  // exception is localhost, which is where the app runs and is not resolvable
  // from the build — exempt that specific host rather than disabling the check.
  ignoreDeadLinks: [/^https?:\/\/localhost/],

  sitemap: {
    hostname: ORIGIN + '/',
  },

  head: [
    ['link', { rel: 'icon', href: BASE + 'favicon.svg', type: 'image/svg+xml' }],
    ['meta', { name: 'theme-color', content: '#4f46e5' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'pdf-ocr' }],
    ['meta', { property: 'og:title', content: 'pdf-ocr — scanned PDFs to editable text' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'Self-hosted Flask web app that turns scanned PDFs into DOCX, TXT, Markdown or HTML using OCR.',
      },
    ],
    ['meta', { property: 'og:url', content: ORIGIN + '/' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
    ['script', { type: 'application/ld+json' }, JSON.stringify(structuredData)],
  ],

  themeConfig: {
    outline: [2, 3],

    nav: [
      { text: 'Guide', link: '/guide/introduction', activeMatch: '/guide/' },
      { text: 'Reference', link: '/reference/configuration', activeMatch: '/reference/' },
      { text: 'Security', link: '/security' },
      {
        text: 'v0.5.1',
        items: [
          { text: 'Releases', link: 'https://github.com/fabriziosalmi/pdf-ocr/releases' },
          { text: 'Container image', link: 'https://github.com/fabriziosalmi/pdf-ocr/pkgs/container/pdf-ocr' },
          { text: 'Contributing', link: 'https://github.com/fabriziosalmi/pdf-ocr/blob/main/CONTRIBUTING.md' },
        ],
      },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Introduction', link: '/guide/introduction' },
            { text: 'Quickstart', link: '/guide/quickstart' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'OCR engines', link: '/guide/engines' },
            { text: 'Deployment', link: '/guide/deployment' },
            { text: 'Troubleshooting', link: '/guide/troubleshooting' },
          ],
        },
      ],
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Configuration', link: '/reference/configuration' },
            { text: 'HTTP endpoints', link: '/reference/endpoints' },
            { text: 'How a conversion works', link: '/reference/pipeline' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/fabriziosalmi/pdf-ocr' },
    ],

    editLink: {
      pattern: 'https://github.com/fabriziosalmi/pdf-ocr/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },

    search: { provider: 'local' },

    footer: {
      message:
        'MIT licensed. <a href="/pdf-ocr/privacy">Privacy</a> · <a href="/pdf-ocr/security">Security</a> · <a href="/pdf-ocr/llms.txt">llms.txt</a>',
      copyright: 'Copyright © 2025–2026 Fabrizio Salmi',
    },
  },
})
