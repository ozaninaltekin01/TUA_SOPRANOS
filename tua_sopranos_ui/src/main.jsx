import React from 'react'
import ReactDOM from 'react-dom/client'
import { Ion } from 'cesium'
import App from './App'
import './index.css'

// Cesium Ion token — optional, falls back to bundled Natural Earth imagery
const token = import.meta.env.VITE_CESIUM_TOKEN
if (token && token.length > 10) {
  Ion.defaultAccessToken = token
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
