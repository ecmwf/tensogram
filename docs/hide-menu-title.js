(function () {
    function markIntroductionPage() {
        if (!document.querySelector("main > h1#introduction")) {
            return;
        }

        document.documentElement.classList.add("introduction-page");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", markIntroductionPage, { once: true });
    } else {
        markIntroductionPage();
    }
})();
