"""
Basit Flask Dashboard
=====================

Bu dosya zorunlu değil; demo sırasında son kayıtları ağ üzerinden göstermek için
eklenmiştir. Tasarım özellikle sade tutuldu, çünkü ana amaç inference zincirini
desteklemek; burada tam bir web ürün geliştirmiyoruz.
"""

from __future__ import annotations

import os
from flask import Flask, jsonify, render_template_string, send_from_directory

import logger


app = Flask(__name__)


HTML_TEMPLATE = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Plant Disease Dashboard</title>
  <style>
    body {
      font-family: "Segoe UI", sans-serif;
      background: #f3f7f0;
      color: #17311f;
      margin: 0;
      padding: 32px;
    }
    h1 {
      margin: 0 0 16px 0;
    }
    .card {
      background: white;
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #e3eadf;
    }
    .ok {
      color: #157347;
      font-weight: 600;
    }
    .bad {
      color: #b42318;
      font-weight: 600;
    }
    .thumb {
      width: 80px;
      height: 60px;
      object-fit: cover;
      border-radius: 6px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Son Tespitler</h1>
    <table>
      <thead>
        <tr>
          <th>Görsel</th>
          <th>Zaman</th>
          <th>Sınıf</th>
          <th>Güven</th>
          <th>Durum</th>
        </tr>
      </thead>
      <tbody>
        {% for event in events %}
        <tr>
          <td>
            {% if event.image_path %}
              <img class="thumb" src="/images/{{ event.image_path | basename }}" alt="">
            {% else %}
              —
            {% endif %}
          </td>
          <td>{{ event.timestamp }}</td>
          <td>{{ event.class_name }}</td>
          <td>{{ "%.1f%%" | format(event.confidence * 100) }}</td>
          <td class="{{ 'bad' if event.is_diseased else 'ok' }}">
            {{ 'Hastalıklı' if event.is_diseased else 'Sağlıklı' }}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


@app.template_filter("basename")
def basename_filter(path):
    return os.path.basename(path) if path else ""


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(logger.IMG_DIR, filename)


@app.route("/")
def index():
    events = logger.recent_events(limit=50)
    return render_template_string(HTML_TEMPLATE, events=events)


@app.route("/api/events")
def api_events():
    return jsonify(logger.recent_events(limit=100))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
