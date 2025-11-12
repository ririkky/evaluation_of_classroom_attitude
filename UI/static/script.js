// 表＋名前の表示
document.getElementById("showTableBtn").addEventListener("click", () => {
  // 変更点: /api/images にアクセス (GETリクエスト)
  fetch("/api/images")
    .then(response => response.json())
    .then(data => {
      // data は [{url: '...', name: '...'}, ...] という配列
      if (data.error) {
        document.getElementById("result").innerText = data.error;
      } else {
        // JSONデータからHTMLテーブルを動的に構築
        let tableHtml = '<table><thead><tr><th>画像</th><th>名前</th></tr></thead><tbody>';
        data.forEach(item => {
          tableHtml += `<tr>
                          <td><img src="${item.url}" alt="${item.name}" style="width:100px;"></td>
                          <td>${item.name}</td>
                        </tr>`;
        });
        tableHtml += '</tbody></table>';
        document.getElementById("result").innerHTML = tableHtml;
      }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        document.getElementById("result").innerText = 'データの取得に失敗しました。';
    });
});

// 写真だけ表示（クリックで拡大対応）
document.getElementById("showPhotosBtn").addEventListener("click", () => {
  // 変更点: /api/images にアクセス (GETリクエスト)
  fetch("/api/images")
    .then(response => response.json())
    .then(data => {
      const result = document.getElementById("result");
      if (data.error) {
        result.innerText = data.error;
      } else {
        // JSONデータから画像のHTMLを動的に構築
        let photosHtml = '';
        data.forEach(item => {
          photosHtml += `<img src="${item.url}" alt="${item.name}" class="thumbnail" style="width:150px; margin:5px; cursor:pointer;">`;
        });
        result.innerHTML = photosHtml;

        // --- ここからは元のJSと同じ（モーダル処理） ---
        // 拡大表示用のイベントを付与
        const modal = document.getElementById("myModal");
        const modalImg = document.getElementById("modalImg");
        const closeBtn = document.getElementsByClassName("close")[0];

        document.querySelectorAll(".thumbnail").forEach(img => {
          img.addEventListener("click", () => {
            modal.style.display = "block";
            modalImg.src = img.src;
          });
        });

        closeBtn.onclick = () => (modal.style.display = "none");
        modal.onclick = () => (modal.style.display = "none");
      }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        document.getElementById("result").innerText = 'データの取得に失敗しました。';
    });
});