// ---------- 共通: モーダル制御 ----------
function attachModalHandlers() {
  const modal = document.getElementById("myModal");
  const modalImg = document.getElementById("modalImg");
  const closeBtn = document.getElementsByClassName("close")[0];

  // 画像クリックで拡大
  document.querySelectorAll(".thumbnail").forEach(img => {
    img.addEventListener("click", () => {
      modal.style.display = "block";
      modalImg.src = img.src;
    });
  });

  // 閉じる
  closeBtn.onclick = () => (modal.style.display = "none");
  modal.onclick = () => (modal.style.display = "none");
}

// ---------- 表＋スコアの表示 ----------
document.getElementById("showTableBtn").addEventListener("click", () => {
  fetch("/api/images")
    .then(response => response.json())
    .then(data => {
      const result = document.getElementById("result");

      if (data.error) {
        result.innerText = data.error;
        return;
      }

      // 画像＋各スコアをテーブル形式で構築
      let tableHtml = `
        <table>
          <thead>
            <tr>
              <th>画像</th>
              <th>メッシュID</th>
              <th>総合</th>
              <th>Pitch</th>
              <th>Yaw</th>
              <th>EAR</th>
            </tr>
          </thead>
          <tbody>
      `;

      data.forEach(item => {
        tableHtml += `
          <tr>
            <td>
              <img src="${item.url}" 
                   alt="${item.mesh_id}" 
                   class="thumbnail" 
                   style="width:150px; cursor:pointer;">
            </td>
            <td>${item.mesh_id}</td>
            <td>${item.total_score}</td>
            <td>${item.pitch_score}</td>
            <td>${item.yaw_score}</td>
            <td>${item.ear_score}</td>
          </tr>
        `;
      });

      tableHtml += '</tbody></table>';
      result.innerHTML = tableHtml;

      // テーブル内の画像にもモーダル処理を付与
      attachModalHandlers();
    })
    .catch(error => {
      console.error('Fetch error:', error);
      document.getElementById("result").innerText = 'データの取得に失敗しました。';
    });
});

// ---------- 写真だけ表示（クリックで拡大） ----------
document.getElementById("showPhotosBtn").addEventListener("click", () => {
  fetch("/api/images")
    .then(response => response.json())
    .then(data => {
      const result = document.getElementById("result");

      if (data.error) {
        result.innerText = data.error;
        return;
      }

      let photosHtml = '';
      data.forEach(item => {
        photosHtml += `
          <img src="${item.url}" 
               alt="${item.mesh_id}" 
               class="thumbnail" 
               style="width:150px; margin:5px; cursor:pointer;">
        `;
      });

      result.innerHTML = photosHtml;

      // サムネイルにモーダル処理を付与
      attachModalHandlers();
    })
    .catch(error => {
      console.error('Fetch error:', error);
      document.getElementById("result").innerText = 'データの取得に失敗しました。';
    });
});
