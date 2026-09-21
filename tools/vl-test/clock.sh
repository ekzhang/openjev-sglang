#!/usr/bin/env bash
# 一行式：时钟图片四选一
IMG=/home/alex/图片/clock.png
B64=$(base64 -w0 "$IMG")
curl -sS --max-time 300 http://125.67.215.17:33027/v1/systemone \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"jev-latest\",
       \"state\":[{\"role\":\"user\",\"content\":[
         {\"type\":\"text\",\"text\":\"This image shows a clock face. Read the time.\"},
         {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}]}],
       \"questions\":{\"time\":{\"type\":\"choice\",
         \"instructions\":\"What time does the clock show?\",
         \"criteria\":{\"A\":\"10:50\",\"B\":\"12:30\",\"C\":\"2:25\",\"D\":\"5:40\"}}}}" \
| python3 -m json.tool
