import { defineConfig, type HeadConfig } from 'vitepress'

// Project page on GitHub Pages: https://fabriziosalmi.github.io/pdf-ocr/
const SITE = 'https://fabriziosalmi.github.io'
const BASE = '/pdf-ocr/'
const ORIGIN = SITE + BASE.replace(/\/$/, '')

// La social preview caricata sul repository. GitHub la serve dalla sua CDN, quindi
// non c'e' un PNG da tenere allineato dentro docs/public.
const OG_IMAGE =
  'https://repository-images.githubusercontent.com/966230846/7da7a60a-d947-408d-8bd7-da9935a81bbc'

/**
 * URL assoluto di una pagina. Il sito gira con cleanUrls, quindi niente estensione
 * .html: altrimenti il canonical direbbe una cosa e la sitemap un'altra.
 */
function canonicalFor(relativePath: string): string {
  const path = relativePath.replace(/index\.md$/, '').replace(/\.md$/, '')
  return ORIGIN + '/' + path
}

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
  // from the build, so exempt that specific host rather than disabling the check.
  ignoreDeadLinks: [/^https?:\/\/localhost/],

  sitemap: {
    hostname: ORIGIN + '/',
  },

  head: [
    ['link', { rel: 'icon', href: BASE + 'favicon.svg', type: 'image/svg+xml' }],
    ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: BASE + 'apple-touch-icon.png' }],
    ['meta', { name: 'theme-color', content: '#4f46e5' }],
    ['meta', { name: 'author', content: 'Fabrizio Salmi' }],
    ['meta', { name: 'robots', content: 'index, follow, max-image-preview:large' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'pdf-ocr' }],
    ['meta', { property: 'og:title', content: 'pdf-ocr: scanned PDFs to editable text' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'Self-hosted Flask web app that turns scanned PDFs into DOCX, TXT, Markdown or HTML using OCR.',
      },
    ],
    ['meta', { property: 'og:locale', content: 'en_GB' }],
    ['meta', { property: 'og:image', content: OG_IMAGE }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:image', content: OG_IMAGE }],
    ['script', { type: 'application/ld+json' }, JSON.stringify(structuredData)],
  ],

  // Canonical e og:url per pagina. Senza, ogni pagina dichiarerebbe la radice del
  // sito, che e' esattamente cio' che produce gli avvisi di contenuto duplicato e
  // le anteprime di condivisione sbagliate.
  transformPageData(pageData) {
    const url = canonicalFor(pageData.relativePath)
    const head: HeadConfig[] = pageData.frontmatter.head ?? []
    head.push(
      ['link', { rel: 'canonical', href: url }],
      ['meta', { property: 'og:url', content: url }],
    )
    pageData.frontmatter.head = head
  },

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
