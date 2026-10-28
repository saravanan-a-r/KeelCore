import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../styles/pages.css';
import { LicenseApp } from './LicenseApp';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LicenseApp />
  </StrictMode>,
);
