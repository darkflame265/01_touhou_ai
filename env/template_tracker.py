# env/template_tracker.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from typing import List, Tuple, Optional


@dataclass
class TrackResult:
    x: int
    y: int
    conf: float
    found: bool
    method: str  # "tmpl"


class MultiTemplateTracker:
    """
    Multi-template edge-based tracker.

    개선 핵심(이번 패치):
    - 각 템플릿/스케일마다 minMaxLoc 1개만 뽑지 않고
      matchTemplate 결과에서 Top-K local peak 후보를 뽑는다.
    - 후보별 2차 검증: 색 대신 "엣지 겹침(edge overlap)"로 후보를 필터링/가중한다.
      => best는 더 강해지고 second는 내려가 margin이 커지는 방향.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        template_paths: List[str],
        init_xy: tuple[int, int] | None = None,

        ema_alpha: float = 0.35,
        base_search_radius: int = 260,
        scales: tuple[float, ...] = (0.95, 0.97, 1.0, 1.03, 1.05),

        # ===== 점수 기준 =====
        min_score: float = 0.42,
        min_margin: float = 0.02,

        method: int = cv2.TM_CCOEFF_NORMED,

        # ===== 색 게이트(이번 패치에선 사용 안 함. 디버그용으로 남겨둠) =====
        red_min_ratio: float = 0.06,
        white_min_ratio: float = 0.09,

        # ===== 소프트 트래킹 =====
        soft_update_score: float = 0.25,
        soft_alpha: float = 0.12,

        # ===== 재획득 / 안정화 =====
        strong_score: float = 0.55,
        max_jump: int = 90,

        # ===== vote =====
        vote_radius: int = 18,
        vote_min: int = 2,
        vote_min_score: float = 0.30,

        # ===== ignore(아이템) =====
        ignore_template_paths: Optional[List[str]] = None,
        ignore_min_score: float = 0.50,
        ignore_block_radius: int = 28,
        enable_ignore_block: bool = True,

        # ===== full-search (현 코드에서는 동작 로직 없음, 유지) =====
        enable_full_search: bool = True,
        full_search_after_miss: int = 6,
        full_search_frames: int = 8,
        require_confirm_to_accept: bool = True,

        # ===== NEW: Top-K peak + 2차 검증 파라미터 =====
        peak_k: int = 6,                 # 템플릿/스케일당 local peak 몇 개 볼지
        peak_nms_ksize: int = 9,         # local peak 찾기용 dilate 커널 크기(홀수 권장)
        peak_min_separation: int = 6,    # peak들끼리 너무 붙으면 제거(픽셀)
        edge_overlap_min: float = 0.08,  # 2차 검증: 엣지 겹침 최소 비율 (0.05~0.15 튜닝)
        edge_overlap_weight: float = 0.35,  # 최종 점수에 overlap을 얼마나 섞을지 (0~0.6)
        max_candidates_total: int = 220, # 전체 cand_list 상한(속도/안정)
    ):
        self.w = frame_w
        self.h = frame_h

        self.ema_alpha = float(ema_alpha)
        self.base_r = int(base_search_radius)
        self.scales = tuple(scales)

        self.min_score = float(min_score)
        self.min_margin = float(min_margin)

        self.method = method

        self.red_min_ratio = float(red_min_ratio)
        self.white_min_ratio = float(white_min_ratio)

        self.soft_update_score = float(soft_update_score)
        self.soft_alpha = float(soft_alpha)

        self.strong_score = float(strong_score)
        self.max_jump = int(max_jump)

        self.vote_radius = int(vote_radius)
        self.vote_min = int(vote_min)
        self.vote_min_score = float(vote_min_score)

        self.ignore_min_score = float(ignore_min_score)
        self.ignore_block_radius = int(ignore_block_radius)
        self.enable_ignore_block = bool(enable_ignore_block)

        self.enable_full_search = bool(enable_full_search)
        self.full_search_after_miss = int(full_search_after_miss)
        self.full_search_frames = int(full_search_frames)
        self.require_confirm_to_accept = bool(require_confirm_to_accept)
        self._full_search_left = 0

        # NEW
        self.peak_k = int(max(1, peak_k))
        self.peak_nms_ksize = int(peak_nms_ksize if peak_nms_ksize >= 3 else 3)
        if self.peak_nms_ksize % 2 == 0:
            self.peak_nms_ksize += 1
        self.peak_min_separation = int(max(0, peak_min_separation))
        self.edge_overlap_min = float(edge_overlap_min)
        self.edge_overlap_weight = float(edge_overlap_weight)
        self.max_candidates_total = int(max(30, max_candidates_total))

        if not template_paths:
            raise ValueError("template_paths must be non-empty")

        # ===== 템플릿 로드 (edge) =====
        self.templates: List[Tuple[str, np.ndarray]] = []
        for p in template_paths:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(p)
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (3, 3), 0)
            edge = cv2.Canny(g, 60, 160)
            self.templates.append((p, edge))

        # ===== ignore 템플릿 로드 =====
        self.ignore_templates: List[Tuple[str, np.ndarray]] = []
        if ignore_template_paths:
            for p in ignore_template_paths:
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(p)
                g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                g = cv2.GaussianBlur(g, (3, 3), 0)
                edge = cv2.Canny(g, 60, 160)
                self.ignore_templates.append((p, edge))

        if init_xy is None:
            init_xy = (frame_w // 2, int(frame_h * 0.78))

        self.last_x, self.last_y = int(init_xy[0]), int(init_xy[1])
        self.smooth_x, self.smooth_y = float(self.last_x), float(self.last_y)
        self.miss_count = 0

        # ===== 디버그 =====
        self.last_match_box = None
        self.last_match_template = None
        self.last_match_scale = None

        self.dbg_best = None
        self.dbg_second = None
        self.dbg_margin = None
        self.dbg_red = None
        self.dbg_white = None
        self.dbg_reject = None
        self.dbg_candidate_center = None
        self.dbg_confirm = None
        self.dbg_votes = None

        self.dbg_ignore_hit = None
        self.dbg_full_search = False

        self.dbg_similarity = None
        self.dbg_similarity_pct = None

    # -------------------------------------------------

    def _roi_from_last(self, frame: np.ndarray):
        if self.enable_full_search and self._full_search_left > 0:
            return frame, 0, 0

        r = int(self.base_r * (1.0 + min(self.miss_count, 10) * 0.35))
        x0 = max(0, self.last_x - r)
        y0 = max(0, self.last_y - r)
        x1 = min(self.w, self.last_x + r)
        y1 = min(self.h, self.last_y + r)
        return frame[y0:y1, x0:x1], x0, y0

    def _clamp_xy(self, x: float, y: float):
        return (
            max(0, min(self.w - 1, int(x))),
            max(0, min(self.h - 1, int(y))),
        )

    def _soft_follow(self, cx: int, cy: int):
        a = self.soft_alpha
        self.smooth_x = (1 - a) * self.smooth_x + a * cx
        self.smooth_y = (1 - a) * self.smooth_y + a * cy

    # -------------------------------------------------
    # ignore 관련은 기존 유지 (Top-K는 main template에만 적용)
    # -------------------------------------------------

    def _best_ignore_in_roi(self, edge: np.ndarray):
        if not self.ignore_templates:
            return None

        best = None
        best_score = -1e9

        for path, tmpl0 in self.ignore_templates:
            for s in self.scales:
                tmpl = tmpl0
                if s != 1.0:
                    h, w = tmpl0.shape
                    tmpl = cv2.resize(
                        tmpl0,
                        (max(8, int(w * s)), max(8, int(h * s))),
                        interpolation=cv2.INTER_AREA,
                    )

                th, tw = tmpl.shape
                rh, rw = edge.shape
                if rh < th or rw < tw:
                    continue

                res = cv2.matchTemplate(edge, tmpl, self.method)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)
                score = float(maxv)
                if score > best_score:
                    tx, ty = int(maxloc[0]), int(maxloc[1])
                    cx = tx + tw // 2
                    cy = ty + th // 2
                    best_score = score
                    best = (score, cx, cy, tx, ty, tw, th, path, float(s))

        return best

    # -------------------------------------------------
    # NEW: local peak 추출 + edge overlap 2차 검증
    # -------------------------------------------------

    def _pick_topk_peaks(self, res: np.ndarray, k: int, min_score: float):
        """
        matchTemplate 결과 res에서 local maxima를 뽑아서 Top-K 반환.
        return: List[(score, x, y)]  (x,y는 res좌표 = template top-left)
        """
        if res is None or res.size == 0:
            return []

        # local maxima mask
        ksize = self.peak_nms_ksize
        kernel = np.ones((ksize, ksize), np.uint8)
        dil = cv2.dilate(res, kernel)

        # res가 dil과 같고, threshold 이상인 점만 후보
        mask = (res >= float(min_score)) & (res == dil)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []

        scores = res[ys, xs].astype(np.float32)

        # score 내림차순 Top-K
        idx = np.argsort(-scores)
        picked = []
        sep2 = float(self.peak_min_separation * self.peak_min_separation)

        for ii in idx:
            sx = int(xs[ii])
            sy = int(ys[ii])
            sv = float(scores[ii])

            # 너무 가까운 peak 제거(옵션)
            if self.peak_min_separation > 0 and picked:
                ok = True
                for (_, px, py) in picked:
                    dx = sx - px
                    dy = sy - py
                    if (dx * dx + dy * dy) <= sep2:
                        ok = False
                        break
                if not ok:
                    continue

            picked.append((sv, sx, sy))
            if len(picked) >= k:
                break

        return picked

    def _edge_overlap(self, edge_roi: np.ndarray, tmpl_edge: np.ndarray, tx: int, ty: int):
        """
        ROI의 edge와 tmpl_edge가 얼마나 겹치는지 (0~1) 비율 반환.
        """
        th, tw = tmpl_edge.shape[:2]
        patch = edge_roi[ty:ty + th, tx:tx + tw]
        if patch.size == 0 or patch.shape[0] != th or patch.shape[1] != tw:
            return 0.0

        # 둘 다 edge(>0)인 픽셀 비율
        inter = cv2.bitwise_and(patch, tmpl_edge)
        inter_n = float(cv2.countNonZero(inter))
        tmpl_n = float(cv2.countNonZero(tmpl_edge)) + 1e-9
        return float(inter_n / tmpl_n)

    def _cluster_vote(self, cand_list: List[Tuple[float, int, int, int, int, int, int, str, float]]):
        if not cand_list:
            return None

        r2 = float(self.vote_radius * self.vote_radius)

        best_votes = 0
        best_sum_score = -1e9
        best_center = None
        best_ref = None
        best_scores_sorted = None

        for it in cand_list:
            score_i, cx_i, cy_i, tx_i, ty_i, tw_i, th_i, tmpl_i, sc_i = it

            votes = 0
            sum_score = 0.0
            sum_x = 0.0
            sum_y = 0.0
            scores = []

            for (score_j, cx_j, cy_j, tx_j, ty_j, tw_j, th_j, tmpl_j, sc_j) in cand_list:
                dx = cx_j - cx_i
                dy = cy_j - cy_i
                if (dx * dx + dy * dy) <= r2:
                    votes += 1
                    sum_score += float(score_j)
                    sum_x += float(cx_j)
                    sum_y += float(cy_j)
                    scores.append(float(score_j))

            if votes <= 0:
                continue

            if (votes > best_votes) or (votes == best_votes and sum_score > best_sum_score):
                best_votes = votes
                best_sum_score = sum_score
                best_center = (int(round(sum_x / votes)), int(round(sum_y / votes)))
                best_ref = (tx_i, ty_i, tw_i, th_i, tmpl_i, sc_i)
                scores.sort(reverse=True)
                best_scores_sorted = scores

        if best_center is None:
            return None

        best_score = best_scores_sorted[0] if best_scores_sorted else -1.0
        second_score = best_scores_sorted[1] if (best_scores_sorted and len(best_scores_sorted) >= 2) else -1.0

        return {
            "votes": best_votes,
            "cx": best_center[0],
            "cy": best_center[1],
            "ref": best_ref,
            "best": float(best_score),
            "second": float(second_score),
        }

    def _best_and_second_in_roi(self, roi_bgr: np.ndarray):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = cv2.Canny(gray, 60, 160)

        # ---- ignore 체크 (기존 방식 유지: 최고점 1개로 center 잡아 block) ----
        self.dbg_ignore_hit = None
        ignore_hit = self._best_ignore_in_roi(edge)
        ignore_center = None
        if ignore_hit is not None:
            sc_ig, cx_ig, cy_ig, tx_ig, ty_ig, tw_ig, th_ig, path_ig, scale_ig = ignore_hit
            if float(sc_ig) >= self.ignore_min_score:
                ignore_center = (int(cx_ig), int(cy_ig))
                self.dbg_ignore_hit = (float(sc_ig), int(cx_ig), int(cy_ig), str(path_ig))

        # ---- 후보 생성 (Top-K peaks) ----
        cand_list = []
        rh, rw = edge.shape[:2]

        for path, tmpl0 in self.templates:
            for s in self.scales:
                tmpl = tmpl0
                if s != 1.0:
                    h, w = tmpl0.shape
                    tmpl = cv2.resize(
                        tmpl0,
                        (max(8, int(w * s)), max(8, int(h * s))),
                        interpolation=cv2.INTER_AREA,
                    )

                th, tw = tmpl.shape
                if rh < th or rw < tw:
                    continue

                res = cv2.matchTemplate(edge, tmpl, self.method)

                # ✅ Top-K local peaks
                peaks = self._pick_topk_peaks(res, self.peak_k, self.vote_min_score)
                if not peaks:
                    continue

                for score, tx, ty in peaks:
                    cx = tx + tw // 2
                    cy = ty + th // 2

                    # ignore block
                    if self.enable_ignore_block and ignore_center is not None:
                        dx = cx - ignore_center[0]
                        dy = cy - ignore_center[1]
                        if (dx * dx + dy * dy) <= float(self.ignore_block_radius * self.ignore_block_radius):
                            continue

                    # ✅ 2차 검증(색 대신): edge overlap
                    ov = self._edge_overlap(edge, tmpl, tx, ty)
                    if ov < self.edge_overlap_min:
                        continue

                    # 최종 점수: template score + overlap 보정(약하게)
                    # overlap이 높을수록 best↑, 애매한 것(요정/탄막)은 overlap에서 걸러지거나 score가 덜 올라감
                    final_score = float(score) * (1.0 - self.edge_overlap_weight + self.edge_overlap_weight * (ov / max(1e-6, self.edge_overlap_min)))
                    # 과도한 폭주 방지
                    final_score = float(max(0.0, min(1.5, final_score)))

                    cand_list.append((final_score, cx, cy, tx, ty, tw, th, path, float(s)))

                    if len(cand_list) >= self.max_candidates_total:
                        break
                if len(cand_list) >= self.max_candidates_total:
                    break
            if len(cand_list) >= self.max_candidates_total:
                break

        if not cand_list:
            return None

        voted = self._cluster_vote(cand_list)
        if voted is None or voted["votes"] < self.vote_min:
            cand_list.sort(key=lambda x: x[0], reverse=True)
            score, cx, cy, tx, ty, tw, th, path, sc = cand_list[0]
            second = cand_list[1][0] if len(cand_list) >= 2 else -1.0
            return float(score), float(second), (tx, ty), (tw, th), path, float(sc), 1

        tx, ty, tw, th, path, sc = voted["ref"]
        best = voted["best"]
        second = voted["second"]
        return float(best), float(second), (int(tx), int(ty)), (int(tw), int(th)), path, float(sc), int(voted["votes"])

    # -------------------------------------------------

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        # reset debug
        self.dbg_reject = None
        self.dbg_red = None
        self.dbg_white = None
        self.dbg_candidate_center = None
        self.dbg_confirm = None
        self.dbg_votes = None
        self.dbg_ignore_hit = None

        self.dbg_similarity = None
        self.dbg_similarity_pct = None

        roi, ox, oy = self._roi_from_last(frame_bgr)

        m = self._best_and_second_in_roi(roi)
        if m is None:
            self.miss_count += 1
            cx, cy = self._clamp_xy(self.smooth_x, self.smooth_y)
            self.dbg_reject = "no_match"
            return TrackResult(cx, cy, -1.0, False, "tmpl")

        best, second, (tx, ty), (tw, th), tmpl_path, sc, votes = m
        margin = best - second

        self.dbg_best = best
        self.dbg_second = second
        self.dbg_margin = margin
        self.dbg_votes = int(votes)

        # similarity (percent)
        try:
            b = float(best)
            b01 = max(0.0, min(1.0, b))
            self.dbg_similarity = b
            self.dbg_similarity_pct = int(b01 * 100.0 + 0.5)
        except Exception:
            self.dbg_similarity = None
            self.dbg_similarity_pct = None

        # 후보 중심(전역좌표)
        cand_x, cand_y = self._clamp_xy(
            ox + tx + tw // 2,
            oy + ty + th // 2,
        )
        self.dbg_candidate_center = (cand_x, cand_y)

        # 디버그: 마지막 매칭 박스 저장(추후 DebugViz 등에서 사용 가능)
        self.last_match_box = (ox + tx, oy + ty, int(tw), int(th))
        self.last_match_template = str(tmpl_path)
        self.last_match_scale = float(sc)

        # jump 허용
        dx = cand_x - int(self.smooth_x)
        dy = cand_y - int(self.smooth_y)
        dist = (dx * dx + dy * dy) ** 0.5
        dynamic_jump = self.max_jump * (1.0 + 0.6 * min(self.miss_count, 10))
        jump_ok = dist <= dynamic_jump

        # score가 너무 낮으면 reject
        if best < self.soft_update_score:
            self.miss_count += 1
            self.dbg_reject = "score"
            cx, cy = self._clamp_xy(self.smooth_x, self.smooth_y)
            return TrackResult(cx, cy, best, False, "tmpl")

        # votes가 충분할 때만 soft_follow
        if jump_ok and (votes >= self.vote_min):
            self._soft_follow(cand_x, cand_y)

        # vote 약하면 reject
        if votes < self.vote_min:
            self.miss_count += 1
            self.dbg_reject = "vote"
            cx, cy = self._clamp_xy(self.smooth_x, self.smooth_y)
            return TrackResult(cx, cy, best, False, "tmpl")

        # margin guard (기존 유지)
        margin_guard = True
        if best >= self.strong_score:
            margin_guard = False
        if votes >= (self.vote_min + 1):
            margin_guard = False

        if margin_guard and (margin < self.min_margin):
            self.miss_count += 1
            self.dbg_reject = f"margin(m={margin:.3f}<min={self.min_margin:.3f})"
            cx, cy = self._clamp_xy(self.smooth_x, self.smooth_y)
            return TrackResult(cx, cy, best, False, "tmpl")

        # ✅ 이번 버전은 색을 신뢰하지 않으니 confirm은 "강한 점수 + 점프 OK" 정도로만 둠
        self.dbg_confirm = bool(best >= self.strong_score and jump_ok)

        # accept
        self.miss_count = 0
        self.last_x, self.last_y = cand_x, cand_y

        a = self.ema_alpha
        self.smooth_x = (1 - a) * self.smooth_x + a * cand_x
        self.smooth_y = (1 - a) * self.smooth_y + a * cand_y

        cx, cy = self._clamp_xy(self.smooth_x, self.smooth_y)
        return TrackResult(cx, cy, best, True, "tmpl")
