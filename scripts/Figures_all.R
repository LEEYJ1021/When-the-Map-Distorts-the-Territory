# ============================================================
# "When the Map Distorts the Territory"
# Figure generation — v6.2  (all overlaps definitively fixed)
#
# Per-figure diagnosis and fix:
#
# Fig1:  "distortion propagates" box overlaps H1 box top edge
#         FIX: move it to y = boxy_top + bh/2 + 0.055 (above both boxes)
#              and shift x slightly right so it clears H1's right edge
#
# Fig2C: JP label overlaps the JP point (large red circle at 6.2, 93.8)
#        DE label sits right next to JP label
#         FIX: JP label -> top-left corner (lx=5, ly=58); clear of point
#              DE label -> far right of plot (lx=52, ly=104); staggered up
#              Segment geometry recalculated to match new label positions
#
# Fig4C: bar1 ICC label clipped by left y-axis; labels still too close
#         FIX: bar1 label at x=1.05 (right side, above bar), ly=1.60
#              bar2 label at x=2.22 (right side), ly=1.75
#              Both on SAME side, staggered vertically (gap=0.15)
#              Arrow endpoints: from label bottom -> bar top (y=1.02)
#              y-axis limit = 1.95
#
# FigA:  Large dead-space gap between upper diagram and ICC bars
#        bar2 label still pokes into diagram area
#         FIX: Use pushViewport with explicit layout splitting the canvas
#              top 55% = diagram, bottom 45% = ICC bars
#              Within ICC viewport, bar geometry uses relative coords [0,1]
#              bar2 label placed to right within ICC viewport only
#
# FigB:  B2: estimate label overlaps point; CI label overlaps error bar
#         FIX: estimate at y=ci_hi+0.008 (above upper CI whisker)
#              CI label at y=ci_lo-0.010 (below lower CI whisker)
#              y-axis limit extended to 0.038 top, -0.130 bottom
# ============================================================

library(ggplot2)
library(gridExtra)
library(grid)
library(scales)

OUT <- file.path(getwd(), "outputs")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

clr <- list(
  jp = "#C0392B", kr = "#2980B9", us = "#27AE60", de = "#8E44AD",
  rc = "#E74C3C", rn = "#2C3E50",
  h1 = "#1A5276", h2 = "#117A65", h3 = "#6C3483",
  agent = "#E67E22", sess = "#2874A6",
  grid = "#ECF0F1", text = "#2C3E50", sub = "#7F8C8D"
)

base_theme <- theme_minimal(base_size = 11) +
  theme(
    text             = element_text(color = clr$text, family = "sans"),
    plot.title       = element_text(size = 12, face = "bold", color = clr$text,
                                    margin = margin(b = 4)),
    plot.subtitle    = element_text(size = 8.5, color = clr$sub,
                                    margin = margin(b = 8)),
    axis.title       = element_text(size = 9, color = clr$sub),
    axis.text        = element_text(size = 8.5, color = clr$text),
    panel.grid.major = element_line(color = clr$grid, linewidth = 0.4),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    plot.margin      = margin(6, 8, 6, 8),
    legend.position  = "none"
  )

# ============================================================
# FIG 1
# FIX: "distortion propagates" was overlapping the H1 box top.
#   New position: y = boxy_top + bh/2 + 0.055 (clearly above both boxes)
#   x = gap_mid (midpoint between H1 right edge and H2 left edge)
#   Add white bg rect so text stays readable against white bg
# ============================================================
cat("Generating Fig 1...\n")

fig1 <- function() {
  grid.newpage()
  pushViewport(viewport(width = 1, height = 1))
  grid.rect(gp = gpar(fill = "white", col = NA))
  
  grid.text(
    "Fig. 1  Causal Architecture: Measurement Bias \u2192 Multi-Agent Discourse Failure",
    x = 0.5, y = 0.965,
    gp = gpar(fontsize = 13, fontface = "bold", col = clr$text)
  )
  grid.text(
    "H1 \u2192 H2 \u2192 H3 causal chain with confirmed effect sizes",
    x = 0.5, y = 0.928,
    gp = gpar(fontsize = 9.5, col = clr$sub)
  )
  
  h1x <- 0.185; h2x <- 0.540; h3x <- 0.540; dbx <- 0.185
  boxy_top <- 0.660; boxy_bot <- 0.285
  bw_wide <- 0.300; bw_narrow <- 0.300; bh <- 0.220; bh_small <- 0.140
  
  draw_box <- function(cx, cy, w, h, t1, t2 = NULL, t3 = NULL,
                       fill, border, fs = 9.5) {
    grid.roundrect(x = cx, y = cy, width = w, height = h, r = unit(7, "pt"),
                   gp = gpar(fill = fill, col = border, lwd = 1.6))
    if (!is.null(t2)) {
      grid.text(t1, x = cx, y = cy + 0.038,
                gp = gpar(fontsize = fs, fontface = "bold", col = "white"))
      grid.text(t2, x = cx, y = cy,
                gp = gpar(fontsize = 8, col = "#FADBD8"))
      if (!is.null(t3))
        grid.text(t3, x = cx, y = cy - 0.040,
                  gp = gpar(fontsize = 7.5, col = "#FADBD8"))
    } else {
      grid.text(t1, x = cx, y = cy,
                gp = gpar(fontsize = fs, fontface = "bold", col = "white"))
    }
  }
  
  draw_box(h1x, boxy_top, bw_wide, bh,
           "H1: Structural MNAR",
           "JP home-filing: 6.2%   US: 91.1%",
           "\u03c7\u00b2(21) = 18,336.51,  V = 0.44",
           fill = clr$h1, border = "#154360")
  draw_box(h2x, boxy_top, bw_narrow, bh,
           "H2: Convergence Suppression",
           "\u039417.6% vs 27.9%  (\u221210.3 pp)",
           "z = \u22123.88,  OR = 0.55,  NNT = 9.7",
           fill = clr$h2, border = "#0B5345")
  draw_box(h3x, boxy_bot, bw_narrow, bh,
           "H3: JP Stance Closure",
           "PM = 35.2% (agent) / 60.3% (session)",
           "indirect = \u22120.029,  z \u2248 \u221215.7",
           fill = clr$h3, border = "#4A235A")
  draw_box(dbx, boxy_bot, bw_wide, bh_small,
           "Patent DB",
           "CPC B60W,  N = 53,199",
           fill = "#566573", border = "#2C3E50", fs = 9)
  
  h1_right <- h1x + bw_wide / 2
  h2_left  <- h2x - bw_narrow / 2
  gap_mid  <- (h1_right + h2_left) / 2
  
  # Arrow between H1 and H2
  grid.lines(x = c(h1_right, h2_left), y = c(boxy_top, boxy_top),
             gp = gpar(col = "#1A5276", lwd = 2.2),
             arrow = arrow(length = unit(9, "pt"), type = "closed", ends = "last"))
  
  # FIX: label ABOVE the arrow line, in the open space between title and boxes
  # boxy_top + bh/2 = top of boxes = 0.660 + 0.110 = 0.770
  # Place label at y = 0.810 (well above box tops, below subtitle at 0.928)
  label_y  <- boxy_top + bh / 2 + 0.060   # = 0.830 approx
  label_x  <- gap_mid                       # midpoint of gap
  
  # Draw arrow from label down to the H1->H2 arrow line
  grid.lines(x = c(label_x, label_x),
             y = c(label_y - 0.022, boxy_top + 0.004),
             gp = gpar(col = "#1A5276", lwd = 1.0, lty = "solid"))
  
  # Label box in open space
  grid.roundrect(x = label_x, y = label_y, width = 0.200, height = 0.038,
                 r = unit(4, "pt"),
                 gp = gpar(fill = "#EAF2FF", col = "#1A5276", lwd = 0.8))
  grid.text("distortion propagates",
            x = label_x, y = label_y,
            gp = gpar(fontsize = 9, col = "#1A5276", fontface = "italic"))
  
  # Dashed vertical from H1 to Patent DB
  grid.lines(x = c(h1x, h1x), y = c(boxy_top - bh / 2, boxy_bot + bh_small / 2),
             gp = gpar(col = "#6C3483", lwd = 1.5, lty = "dashed"))
  grid.text("MNAR\nmechanism",
            x = h1x - 0.100,
            y = (boxy_top - bh / 2 + boxy_bot + bh_small / 2) / 2,
            gp = gpar(fontsize = 8, col = "#6C3483", fontface = "italic"))
  
  # Arrow from H2 to H3
  grid.lines(x = c(h2x, h3x), y = c(boxy_top - bh / 2, boxy_bot + bh / 2),
             gp = gpar(col = clr$h3, lwd = 2),
             arrow = arrow(length = unit(9, "pt"), type = "closed"))
  grid.text("mediated by\nstance closure",
            x = h2x + 0.115,
            y = (boxy_top - bh / 2 + boxy_bot + bh / 2) / 2,
            gp = gpar(fontsize = 8, col = clr$h3, fontface = "italic"))
  
  # ICC diagnostics box
  grid.roundrect(x = 0.860, y = 0.480, width = 0.240, height = 0.270,
                 r = unit(5, "pt"),
                 gp = gpar(fill = "#FDFEFE", col = "#BDC3C7", lwd = 1.2))
  grid.text("ICC diagnostics",
            x = 0.860, y = 0.585,
            gp = gpar(fontsize = 9.5, fontface = "bold", col = clr$text))
  grid.text("converged:   ICC = 1.000",
            x = 0.860, y = 0.530,
            gp = gpar(fontsize = 8.5, col = clr$sub))
  grid.text("probability:  ICC = 0.103",
            x = 0.860, y = 0.485,
            gp = gpar(fontsize = 8.5, col = clr$sub))
  grid.text("\u21921:1 cross-level design",
            x = 0.860, y = 0.436,
            gp = gpar(fontsize = 8, col = "#A93226", fontface = "italic"))
  
  grid.text(
    paste0("Note: Effect sizes from raw-data recompute (s4_simulations.jsonl, N = 1,000 sessions). ",
           "H1 from step3_applicant_groups crosswalk."),
    x = 0.5, y = 0.034,
    gp = gpar(fontsize = 8, col = clr$sub))
  popViewport()
}

png(paste0(OUT, "/fig1_theory_v6.png"), width = 3200, height = 2000, res = 300)
fig1(); dev.off()
cat("  -> fig1_theory_v6.png\n")


# ============================================================
# FIG 2C
# FIX: JP point is at (6.2, 93.8) with large radius (~16 in size units)
#      In plot coords (0-110 scale), point radius ≈ 5-6 units
#      JP label was at (22, 76) -> still overlapping point edge
#      DE label at (44, 91) -> near JP label
#
#   New layout:
#     JP:  label in TOP-LEFT corner at (lx=4, ly=54)
#            - point is at (6.2, 93.8), so ly=54 is far BELOW the point
#            - segment goes UP from label top to point bottom
#     KR:  label at (lx=68, ly=36) - bottom right quadrant (clear of KR point at 49.6,50.4)
#     DE:  label at (lx=52, ly=108) - above and right (clear of JP circle)
#     US:  label at (lx=72, ly=18) - bottom right (clear of US point at 91.1,8.9)
#
#   All segments: start at label edge, end 6 units short of point center
# ============================================================
cat("Generating Fig 2...\n")

df_h1a <- data.frame(
  country = factor(c("US", "KR", "DE", "JP"), levels = c("US", "KR", "DE", "JP")),
  rate    = c(91.1, 49.6, 24.6, 6.2),
  n       = c(9021, 3949, 5581, 13094),
  fill    = c(clr$us, clr$kr, clr$de, clr$jp)
)
p2a <- ggplot(df_h1a, aes(x = country, y = rate, fill = country)) +
  geom_col(width = 0.6, color = "white") +
  geom_text(aes(label = paste0(rate, "%")),
            vjust = -0.5, size = 3.8, fontface = "bold") +
  geom_text(aes(label = paste0("n = ", format(n, big.mark = ","))),
            vjust = -2.4, size = 3.0, color = clr$sub) +
  scale_fill_manual(values = setNames(df_h1a$fill, df_h1a$country)) +
  scale_y_continuous(limits = c(0, 115), labels = function(x) paste0(x, "%")) +
  labs(title = "(A) Home-jurisdiction filing rate",
       subtitle = "\u03c7\u00b2(21) = 18,336.51, p < .001, V = 0.44",
       x = "Nationality", y = "Home-filing rate (%)") +
  base_theme +
  theme(plot.subtitle = element_text(size = 8.5, color = "#A93226", face = "italic"))

df_h1b <- data.frame(
  dest = factor(c("US", "EP", "Other (non-JP)", "JP (home)"),
                levels = c("JP (home)", "Other (non-JP)", "EP", "US")),
  pct  = c(80.8, 9.3, 5.4, 4.5),
  fill = c(clr$us, "#BDC3C7", "#85929E", clr$jp)
)
p2b <- ggplot(df_h1b, aes(x = dest, y = pct, fill = dest)) +
  geom_col(width = 0.6, color = "white") +
  geom_text(aes(label = paste0(pct, "%")), hjust = -0.3, size = 3.6, fontface = "bold") +
  scale_fill_manual(values = setNames(df_h1b$fill, df_h1b$dest)) +
  scale_y_continuous(limits = c(0, 100), labels = function(x) paste0(x, "%")) +
  coord_flip() +
  labs(title = "(B) Toyota Group filing destinations",
       subtitle = "JP home-filing: 4.5%  (269 / 6,016 patents)",
       x = NULL, y = "Share (%)") +
  base_theme

# Panel C
# Points: JP(6.2,93.8), DE(24.6,75.4), KR(49.6,50.4), US(91.1,8.9)
# JP has the largest point (n=13094), radius ~8 units in data space
# Key: JP label goes BELOW the point to avoid overlap with DE
df_h1c <- data.frame(
  id    = c("JP\n(MNAR)", "KR", "DE", "US"),
  home  = c(6.2,  49.6, 24.6, 91.1),
  miss  = c(93.8, 50.4, 75.4,  8.9),
  n_tot = c(13094, 3949, 5581, 9021),
  col   = c(clr$jp, clr$kr, clr$de, clr$us),
  # Label anchors — placed in unoccupied zones
  lx    = c(4,    68,   52,   72),
  ly    = c(54,   36,  108,   20),
  # Segment start (at label edge)
  sx    = c(6,    65,   46,   70),
  sy    = c(57,   39,  105,   21),
  # Segment end (near point edge, 6-7 units from centre)
  ex    = c(6.5,  52,   29,   89),
  ey    = c(87,   53,   79,   12)
)

p2c <- ggplot(df_h1c, aes(x = home, y = miss, color = id)) +
  geom_point(aes(size = n_tot), alpha = 0.85) +
  geom_segment(aes(x = sx, y = sy, xend = ex, yend = ey),
               linewidth = 0.45, alpha = 0.70, show.legend = FALSE) +
  geom_label(aes(x = lx, y = ly, label = id, fill = id),
             color = "white", fontface = "bold", size = 3.1,
             label.padding = unit(0.22, "lines"),
             label.r = unit(0.12, "lines"),
             show.legend = FALSE) +
  scale_color_manual(values = setNames(df_h1c$col, df_h1c$id)) +
  scale_fill_manual(values  = setNames(df_h1c$col, df_h1c$id)) +
  scale_size_continuous(range = c(5, 16), guide = "none") +
  scale_x_continuous(limits = c(0, 110), labels = function(x) paste0(x, "%")) +
  scale_y_continuous(limits = c(0, 115), labels = function(x) paste0(x, "%")) +
  annotate("text", x = 62, y = 65,
           label = "MNAR: missingness\ncorrelated with IP strategy",
           size = 2.9, color = "#A93226", fontface = "italic", hjust = 0.5) +
  labs(title = "(C) Home-filing rate vs. attribution missingness",
       subtitle = "Point size \u221d corpus count;  JP is the MNAR outlier",
       x = "Home-filing rate (%)", y = "Attribution missingness (%)") +
  base_theme

fig2 <- arrangeGrob(
  p2a, p2b, p2c, ncol = 3,
  top = textGrob(
    "Fig. 2  Structural (MNAR) Missingness in Patent-Nationality Mapping",
    gp = gpar(fontsize = 13, fontface = "bold", col = clr$text), y = 0.65),
  bottom = textGrob(
    paste0("Source: step3_applicant_groups.parquet crosswalk (researcher-constructed).",
           "  N = 53,199 citation-validated B60W patents."),
    gp = gpar(fontsize = 8, col = clr$sub), y = 0.45)
)

png(paste0(OUT, "/fig2_H1_mnar_v6.png"), width = 4200, height = 1600, res = 300)
grid.draw(fig2); dev.off()
cat("  -> fig2_H1_mnar_v6.png\n")


# ============================================================
# FIG 4 Panel C
# FIX: bar1 label was clipped by y-axis on the left.
#   New strategy: BOTH labels on the RIGHT side, staggered vertically
#     bar1 (converged, x=1):    label at x=1.28, y=1.62
#     bar2 (stance_entropy, x=2): label at x=2.28, y=1.78
#   Arrows: from label left edge -> bar top (y=1.02)
#   y-axis limit = 1.96
#   bar3/4 labels raised to ICC + 0.24
# ============================================================
cat("Generating Fig 4...\n")

p4a_grob <- function() {
  pushViewport(viewport(width = 1, height = 1))
  grid.rect(gp = gpar(fill = "white", col = NA))
  grid.text("(A) Mediation path diagram", x = 0.5, y = 0.96,
            gp = gpar(fontsize = 10, fontface = "bold", col = clr$text))
  
  grid.roundrect(x = 0.13, y = 0.555, width = 0.22, height = 0.135,
                 r = unit(5, "pt"), gp = gpar(fill = clr$h1, col = "#154360", lwd = 1.2))
  grid.text("X: JP nationality",  x = 0.13, y = 0.568,
            gp = gpar(fontsize = 8.5, col = "white", fontface = "bold"))
  grid.text("(dummy, agent-level)", x = 0.13, y = 0.536,
            gp = gpar(fontsize = 7, col = "#AED6F1"))
  
  grid.roundrect(x = 0.500, y = 0.790, width = 0.360, height = 0.130,
                 r = unit(5, "pt"), gp = gpar(fill = clr$h3, col = "#4A235A", lwd = 1.2))
  grid.text("M: stance_direction_r3", x = 0.500, y = 0.802,
            gp = gpar(fontsize = 8.5, col = "white", fontface = "bold"))
  grid.text("(agent-level, continuous)", x = 0.500, y = 0.770,
            gp = gpar(fontsize = 7, col = "#D7BDE2"))
  
  grid.roundrect(x = 0.870, y = 0.555, width = 0.220, height = 0.135,
                 r = unit(5, "pt"), gp = gpar(fill = clr$h2, col = "#0B5345", lwd = 1.2))
  grid.text("Y: probability",    x = 0.870, y = 0.568,
            gp = gpar(fontsize = 8.5, col = "white", fontface = "bold"))
  grid.text("(agent-level, r3)", x = 0.870, y = 0.536,
            gp = gpar(fontsize = 7, col = "#A9DFBF"))
  
  grid.lines(x = c(0.245, 0.318), y = c(0.608, 0.742),
             gp = gpar(col = clr$h3, lwd = 2),
             arrow = arrow(length = unit(7, "pt"), type = "closed"))
  grid.text("a = \u22120.754***", x = 0.248, y = 0.698,
            gp = gpar(fontsize = 8, col = clr$h3, fontface = "bold"))
  
  grid.lines(x = c(0.682, 0.756), y = c(0.742, 0.608),
             gp = gpar(col = clr$h3, lwd = 2),
             arrow = arrow(length = unit(7, "pt"), type = "closed"))
  grid.text("b = +0.038***", x = 0.750, y = 0.698,
            gp = gpar(fontsize = 8, col = clr$h3, fontface = "bold"))
  
  grid.lines(x = c(0.245, 0.756), y = c(0.548, 0.548),
             gp = gpar(col = "#1A5276", lwd = 1.5, lty = "dashed"),
             arrow = arrow(length = unit(7, "pt"), type = "closed"))
  grid.text("c\u2019 = \u22120.053*** (direct)", x = 0.500, y = 0.476,
            gp = gpar(fontsize = 8, col = "#1A5276", fontface = "italic"))
  
  grid.roundrect(x = 0.500, y = 0.275, width = 0.640, height = 0.200,
                 r = unit(5, "pt"), gp = gpar(fill = "#F9F9F9", col = "#BDC3C7", lwd = 1))
  grid.text("Indirect effect (agent-level primary)", x = 0.500, y = 0.355,
            gp = gpar(fontsize = 9, fontface = "bold", col = clr$text))
  grid.text("indirect = \u22120.0289,  z = \u221215.686***,  95% CI [\u22120.033, \u22120.025]",
            x = 0.500, y = 0.310, gp = gpar(fontsize = 8, col = clr$h3))
  grid.text("PM = 35.2% (agent-level)     |     PM = 60.3% (session-level)",
            x = 0.500, y = 0.260,
            gp = gpar(fontsize = 8, col = "#6C3483", fontface = "bold"))
  popViewport()
}

df_spec <- data.frame(
  spec     = factor(c("Agent-level\n(Y = probability)", "Session-level\n(Y = converged)"),
                    levels = c("Agent-level\n(Y = probability)", "Session-level\n(Y = converged)")),
  indirect = c(-0.02888, -0.06213),
  ci_lo    = c(-0.033, -0.088),
  ci_hi    = c(-0.025, -0.039),
  PM       = c(35.2, 60.3),
  z        = c(-15.686, -4.899),
  fill     = c(clr$agent, clr$sess)
)

p4b <- ggplot(df_spec, aes(x = spec, y = indirect, fill = spec)) +
  geom_col(width = 0.5, color = "white") +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                width = 0.12, linewidth = 0.8, color = clr$text) +
  geom_text(aes(label = paste0("PM = ", PM, "%\nz = ", z)),
            vjust = 1.5, size = 3.1, fontface = "bold", color = "white") +
  scale_fill_manual(values = c(clr$agent, clr$sess)) +
  scale_y_continuous(limits = c(-0.115, 0.010)) +
  labs(title = "(B) Indirect effect by specification",
       subtitle = "Both negative & significant; PM is level-dependent",
       x = NULL, y = "Indirect effect") +
  base_theme +
  theme(axis.text.x = element_text(size = 9, face = "bold", lineheight = 1.2))

# Panel C — both ICC=1 labels on RIGHT side of their bars, staggered vertically
p4c <- ggplot(
  data.frame(
    variable = factor(c("converged", "stance_entropy", "probability", "stance_openness"),
                      levels = c("converged", "stance_entropy", "probability", "stance_openness")),
    ICC  = c(1.000, 1.000, 0.103, 0.090),
    fill = c("#C0392B", "#C0392B", "#27AE60", "#27AE60")
  ),
  aes(x = variable, y = ICC, fill = fill)
) +
  geom_col(width = 0.55, color = "white") +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = clr$sub, linewidth = 0.6) +
  
  # bar1 (converged, x=1): label RIGHT of bar at x=1.32, y=1.63
  annotate("label", x = 1.32, y = 1.63,
           label = "ICC = 1.000\nN_eff = 1,000",
           size = 2.85, color = "#C0392B", fill = "#FEF9F9",
           fontface = "bold", label.padding = unit(0.18, "lines"),
           label.r = unit(0.10, "lines"), label.size = 0.3, hjust = 0) +
  # arrow from label towards bar1 top
  annotate("segment",
           x = 1.30, xend = 1.01,
           y = 1.56,  yend = 1.04,
           arrow = arrow(length = unit(4, "pt"), type = "open"),
           color = "#C0392B", linewidth = 0.55) +
  
  # bar2 (stance_entropy, x=2): label RIGHT of bar at x=2.32, y=1.79
  annotate("label", x = 2.32, y = 1.79,
           label = "ICC = 1.000\nN_eff = 1,000",
           size = 2.85, color = "#C0392B", fill = "#FEF9F9",
           fontface = "bold", label.padding = unit(0.18, "lines"),
           label.r = unit(0.10, "lines"), label.size = 0.3, hjust = 0) +
  # arrow from label towards bar2 top
  annotate("segment",
           x = 2.30, xend = 2.01,
           y = 1.72,  yend = 1.04,
           arrow = arrow(length = unit(4, "pt"), type = "open"),
           color = "#C0392B", linewidth = 0.55) +
  
  # bar3 (probability, ICC=0.103): label well above bar
  annotate("text", x = 3, y = 0.103 + 0.24,
           label = "ICC = 0.103\nN_eff = 3,057",
           size = 2.85, color = "#27AE60", fontface = "bold", hjust = 0.5) +
  
  # bar4 (stance_openness, ICC=0.090): label well above bar
  annotate("text", x = 4, y = 0.090 + 0.24,
           label = "ICC = 0.090\nN_eff = 3,148",
           size = 2.85, color = "#27AE60", fontface = "bold", hjust = 0.5) +
  
  # threshold label on the left to avoid bar3/4 label zone
  annotate("text", x = 0.65, y = 0.142,
           label = "threshold", size = 2.7, color = clr$sub,
           fontface = "italic", hjust = 0) +
  
  scale_fill_identity() +
  scale_x_discrete(expand = expansion(mult = c(0.05, 0.38))) +  # extra right margin for labels
  scale_y_continuous(limits = c(0, 1.96)) +
  labs(title = "(C) ICC diagnostics",
       subtitle = "Red = session-shared  (agent-level model inappropriate)",
       x = NULL, y = "Intraclass Correlation (ICC)") +
  base_theme +
  theme(axis.text.x = element_text(size = 8.5))

png(paste0(OUT, "/fig4_H3_mediation_v6.png"), width = 4400, height = 1900, res = 300)
grid.newpage()
pushViewport(viewport(layout = grid.layout(
  2, 3,
  widths  = unit(c(1.15, 0.90, 0.95), "null"),
  heights = unit(c(0.07, 0.93), "null")
)))
pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1:3))
grid.text("Fig. 4  Partial Mediation: JP Stance Closure (Cross-Level Design)",
          x = 0.5, y = 0.5,
          gp = gpar(fontsize = 13, fontface = "bold", col = clr$text))
popViewport()
pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1))
p4a_grob(); popViewport()
pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 2))
print(p4b, vp = viewport(x = 0.5, y = 0.5, width = 0.95, height = 0.92))
popViewport()
pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 3))
print(p4c, vp = viewport(x = 0.5, y = 0.5, width = 0.95, height = 0.92))
popViewport()
dev.off()
cat("  -> fig4_H3_mediation_v6.png\n")


# ============================================================
# FIG 5 — unchanged
# ============================================================
cat("Generating Fig 5...\n")

df_f5_h1 <- data.frame(
  ctry = factor(c("US", "KR", "DE", "JP"), levels = c("US", "KR", "DE", "JP")),
  rate = c(91.1, 49.6, 24.6, 6.2),
  col  = c(clr$us, clr$kr, clr$de, clr$jp)
)
p5_h1 <- ggplot(df_f5_h1, aes(x = ctry, y = rate, fill = ctry)) +
  geom_col(width = 0.6, color = "white") +
  geom_text(aes(label = paste0(rate, "%")), vjust = -0.5, size = 3.4, fontface = "bold") +
  scale_fill_manual(values = setNames(df_f5_h1$col, df_f5_h1$ctry)) +
  scale_y_continuous(limits = c(0, 112), labels = function(x) paste0(x, "%")) +
  labs(title = "(5a) H1: Home-filing rate by nationality",
       subtitle = "\u03c7\u00b2(21) = 18,336.51,  V = 0.44", x = NULL, y = "%") +
  base_theme

df_f5_h1b <- data.frame(
  grp = factor(c("Cram\u00e9r's V", "JP\u2013US gap (pp)"),
               levels = c("Cram\u00e9r's V", "JP\u2013US gap (pp)")),
  val = c(0.44, 84.9), col = c(clr$h1, clr$jp)
)
p5_h1b <- ggplot(df_f5_h1b, aes(x = grp, y = val, fill = grp)) +
  geom_col(width = 0.45, color = "white") +
  geom_text(aes(label = val), vjust = -0.6, size = 3.7, fontface = "bold") +
  scale_fill_manual(values = c(clr$h1, clr$jp)) +
  scale_y_continuous(limits = c(0, 112)) +
  labs(title = "(5b) H1: Effect magnitude",
       subtitle = "Gap = 84.9 pp;  V = 0.44 (large effect)", x = NULL, y = "Value") +
  base_theme

df_f5_h2 <- data.frame(
  cond = factor(c("Baseline\n(reversal_not_confirmed)",
                  "Distortion salient\n(reversal_confirmed)"),
                levels = c("Baseline\n(reversal_not_confirmed)",
                           "Distortion salient\n(reversal_confirmed)")),
  rate = c(27.93, 17.62), fill = c(clr$rn, clr$rc)
)
p5_h2 <- ggplot(df_f5_h2, aes(x = cond, y = rate, fill = cond)) +
  geom_col(width = 0.5, color = "white") +
  geom_text(aes(label = paste0(rate, "%")), vjust = -0.5, size = 3.6, fontface = "bold") +
  annotate("text", x = 1.5, y = 39,
           label = "\u221210.3 pp  (z = \u22123.88, OR = 0.55, NNT = 9.7)",
           size = 3.0, color = clr$rc, fontface = "bold", hjust = 0.5) +
  annotate("segment", x = 1, xend = 2, y = 35.5, yend = 35.5,
           arrow = arrow(ends = "both", length = unit(5, "pt")),
           color = clr$text, linewidth = 0.7) +
  scale_fill_manual(values = c(clr$rn, clr$rc)) +
  scale_y_continuous(limits = c(0, 46), labels = function(x) paste0(x, "%")) +
  labs(title = "(5c) H2: Convergence rate by condition",
       subtitle = "Distortion salience suppresses consensus formation", x = NULL, y = "%") +
  base_theme

df_f5_h2b <- data.frame(
  agent = factor(c("JP", "KR", "US", "DE"), levels = c("JP", "KR", "US", "DE")),
  open  = c(-0.461, 0.611, 0.127, 0.299),
  col   = c(clr$jp, clr$kr, clr$us, clr$de)
)
p5_h2b <- ggplot(df_f5_h2b, aes(x = agent, y = open, fill = agent)) +
  geom_col(width = 0.55, color = "white") +
  geom_hline(yintercept = 0, linewidth = 0.8, color = clr$text) +
  geom_text(aes(label = round(open, 3)),
            vjust = ifelse(df_f5_h2b$open < 0, 1.6, -0.6), size = 3.5, fontface = "bold") +
  scale_fill_manual(values = setNames(df_f5_h2b$col, df_f5_h2b$agent)) +
  scale_y_continuous(limits = c(-0.75, 0.85)) +
  labs(title = "(5d) H2: Round-3 stance openness by agent",
       subtitle = "JP only agent with negative stance (defensive withdrawal)",
       x = "Agent", y = "Mean stance openness") +
  base_theme

df_f5_h3 <- data.frame(
  spec = factor(c("Agent-level\nPM", "Session-level\nPM"),
                levels = c("Agent-level\nPM", "Session-level\nPM")),
  PM   = c(35.2, 60.3), fill = c(clr$agent, clr$sess)
)
p5_h3 <- ggplot(df_f5_h3, aes(x = spec, y = PM, fill = spec)) +
  geom_col(width = 0.45, color = "white") +
  geom_text(aes(label = paste0(PM, "%")), vjust = -0.5, size = 4.0, fontface = "bold") +
  scale_fill_manual(values = c(clr$agent, clr$sess)) +
  scale_y_continuous(limits = c(0, 80), labels = function(x) paste0(x, "%")) +
  labs(title = "(5e) H3: Proportion mediated by specification",
       subtitle = "Cross-level design: both estimates valid", x = NULL, y = "PM (%)") +
  base_theme

df_f5_h3b <- data.frame(
  spec  = factor(c("Agent\n(z = \u221215.686)", "Session\n(z = \u22124.899)"),
                 levels = c("Agent\n(z = \u221215.686)", "Session\n(z = \u22124.899)")),
  ind   = c(-0.02888, -0.06213),
  ci_lo = c(-0.033, -0.088), ci_hi = c(-0.025, -0.039),
  fill  = c(clr$agent, clr$sess)
)
p5_h3b <- ggplot(df_f5_h3b, aes(x = spec, y = ind, fill = spec)) +
  geom_col(width = 0.45, color = "white") +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.1, linewidth = 0.9) +
  geom_hline(yintercept = 0, linewidth = 0.8, color = clr$text) +
  scale_fill_manual(values = c(clr$agent, clr$sess)) +
  scale_y_continuous(limits = c(-0.115, 0.014)) +
  labs(title = "(5f) H3: Indirect effect + 95% CI",
       subtitle = "Consistent direction across both specifications",
       x = NULL, y = "Indirect effect") +
  base_theme

fig5 <- arrangeGrob(
  p5_h1, p5_h1b, p5_h2, p5_h2b, p5_h3, p5_h3b,
  ncol = 3, nrow = 2,
  top = textGrob(
    "Fig. 5  Integrated Evidence Dashboard: H1 \u2192 H2 \u2192 H3",
    gp = gpar(fontsize = 13, fontface = "bold", col = clr$text), y = 0.65),
  bottom = textGrob(
    paste0("All statistics from raw-data recompute.  N = 1,000 sessions (s4_simulations.jsonl).",
           "  H1: N = 53,199 (step3_applicant_groups crosswalk)."),
    gp = gpar(fontsize = 8, col = clr$sub), y = 0.4)
)

png(paste0(OUT, "/fig5_integrated_dashboard_v6.png"), width = 4800, height = 2800, res = 300)
grid.draw(fig5); dev.off()
cat("  -> fig5_integrated_dashboard_v6.png\n")


# ============================================================
# FIG A — complete redesign using explicit viewport split
#
# Strategy: use pushViewport with two sub-viewports:
#   vp_diag: top 54% of canvas (y=0.46..1.00) — diagram section
#   vp_icc:  bottom 44% of canvas (y=0.02..0.46) — ICC bar section
#
# Within vp_icc, ALL coordinates are relative to that viewport [0,1]
# So bar labels never escape into diagram territory
#
# ICC bar geometry in vp_icc:
#   bar_b   = 0.15  (bottom of bars, leaving room for x-labels)
#   bar_max = 0.50  (height of ICC=1 bar fills 50% of vp_icc height)
#   bar tops at bar_b + bar_max = 0.65 for ICC=1
#   Labels: bar1 RIGHT at lx=0.14+0.07=0.21, ly=0.70
#            bar2 RIGHT at lx=0.38+0.07=0.45, ly=0.82
#            Both well below vp_icc top (1.0) which is y=0.46 canvas
# ============================================================
cat("Generating Fig A...\n")

figA <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = "white", col = NA))
  
  # ── Titles (absolute canvas) ──────────────────────────────────────
  grid.text("Fig. A  Cross-Level (2-1-2) Mediation Architecture",
            x = 0.5, y = 0.976,
            gp = gpar(fontsize = 13, fontface = "bold", col = clr$text))
  grid.text("Why a session-level Y with agent-level X/M requires explicit level separation",
            x = 0.5, y = 0.944,
            gp = gpar(fontsize = 9.5, col = clr$sub))
  
  # ── Upper diagram viewport: y = 0.46 .. 0.930 ────────────────────
  pushViewport(viewport(x = 0.5, y = 0.695, width = 1.0, height = 0.470))
  
  # LEFT: Naive box (centred in top half of this viewport)
  lbox_cx <- 0.230; lbox_cy <- 0.52; lbox_w <- 0.420; lbox_h <- 0.82
  grid.roundrect(x = lbox_cx, y = lbox_cy, width = lbox_w, height = lbox_h,
                 r = unit(8, "pt"),
                 gp = gpar(fill = "#FDEDEC", col = "#E74C3C", lwd = 1.5))
  grid.text("Naive: single-level assumption",
            x = lbox_cx, y = 0.885,
            gp = gpar(fontsize = 10, fontface = "bold", col = "#922B21"))
  grid.text("(incorrect for this design)",
            x = lbox_cx, y = 0.820,
            gp = gpar(fontsize = 8.5, col = "#E74C3C", fontface = "italic"))
  
  dmini <- function(x, y, lbl, cf) {
    grid.roundrect(x = x, y = y, width = 0.105, height = 0.230,
                   r = unit(4, "pt"), gp = gpar(fill = cf, col = "white", lwd = 1))
    grid.text(lbl, x = x, y = y,
              gp = gpar(fontsize = 9, col = "white", fontface = "bold"))
  }
  node_y <- 0.490
  dmini(0.085, node_y, "X\n(jp)",     clr$h1)
  dmini(0.230, node_y, "M\n(stance)", clr$h3)
  dmini(0.375, node_y, "Y\n(conv.)",  "#E74C3C")
  
  grid.lines(x = c(0.140, 0.175), y = c(node_y, node_y),
             gp = gpar(col = clr$h3, lwd = 1.5),
             arrow = arrow(length = unit(5, "pt"), type = "closed"))
  grid.lines(x = c(0.285, 0.320), y = c(node_y, node_y),
             gp = gpar(col = clr$h3, lwd = 1.5),
             arrow = arrow(length = unit(5, "pt"), type = "closed"))
  
  grid.text("\u26a0 ICC(Y) = 1.000  \u2192  all within-session variance = 0",
            x = lbox_cx, y = 0.295,
            gp = gpar(fontsize = 8, col = "#922B21", fontface = "bold"))
  grid.text("\u2192 agent-level model degenerates (fixed effects \u2248 0)",
            x = lbox_cx, y = 0.210,
            gp = gpar(fontsize = 8, col = "#922B21"))
  
  # RIGHT: Correct box
  rbox_cx <- 0.755; rbox_cy <- 0.52; rbox_w <- 0.450; rbox_h <- 0.82
  grid.roundrect(x = rbox_cx, y = rbox_cy, width = rbox_w, height = rbox_h,
                 r = unit(8, "pt"),
                 gp = gpar(fill = "#EAFAF1", col = "#27AE60", lwd = 1.5))
  grid.text("Correct: 2-1-2 cross-level",
            x = rbox_cx, y = 0.885,
            gp = gpar(fontsize = 10, fontface = "bold", col = "#1E8449"))
  grid.text("(Preacher, Zyphur & Zhang, 2010)",
            x = rbox_cx, y = 0.820,
            gp = gpar(fontsize = 8.5, col = "#27AE60", fontface = "italic"))
  
  grid.roundrect(x = rbox_cx, y = 0.660, width = 0.390, height = 0.190,
                 r = unit(4, "pt"),
                 gp = gpar(fill = "#D5F5E3", col = "#27AE60", lwd = 0.8))
  grid.text("Level 2 (session):  Y = converged  [ICC = 1.000]",
            x = rbox_cx, y = 0.660,
            gp = gpar(fontsize = 8, col = "#1E8449", fontface = "bold"))
  
  grid.roundrect(x = rbox_cx, y = 0.440, width = 0.390, height = 0.190,
                 r = unit(4, "pt"),
                 gp = gpar(fill = "#EBF5FB", col = "#2980B9", lwd = 0.8))
  grid.text("Level 1 (agent):  X = jp,  M = stance  [ICC \u2248 0.103]",
            x = rbox_cx, y = 0.440,
            gp = gpar(fontsize = 8, col = "#1A5276", fontface = "bold"))
  
  grid.text("\u2713 Agent-level:   indirect = \u22120.029,  PM = 35.2%",
            x = rbox_cx, y = 0.280,
            gp = gpar(fontsize = 8.5, col = "#1E8449", fontface = "bold"))
  grid.text("\u2713 Session-level: indirect = \u22120.062,  PM = 60.3%",
            x = rbox_cx, y = 0.200,
            gp = gpar(fontsize = 8.5, col = "#2980B9", fontface = "bold"))
  
  # Arrow between boxes
  grid.lines(x = c(0.462, 0.528), y = c(0.520, 0.520),
             gp = gpar(col = clr$text, lwd = 2.2),
             arrow = arrow(length = unit(9, "pt"), type = "closed"))
  grid.text("correct\napproach", x = 0.495, y = 0.600,
            gp = gpar(fontsize = 8, col = clr$text, fontface = "italic"))
  
  popViewport()  # end diagram viewport
  
  # ── Divider line ──────────────────────────────────────────────────
  grid.lines(x = c(0.04, 0.96), y = c(0.455, 0.455),
             gp = gpar(col = "#DEDEDE", lwd = 0.8, lty = "solid"))
  
  # ── ICC bar viewport: y = 0.02 .. 0.450 ──────────────────────────
  pushViewport(viewport(x = 0.5, y = 0.225, width = 0.96, height = 0.410))
  
  # Section title
  grid.text("ICC diagnostics by variable",
            x = 0.500, y = 0.960,
            gp = gpar(fontsize = 11, fontface = "bold", col = clr$text))
  
  # Bar geometry (all in vp_icc coords [0,1])
  bar_b   <- 0.140   # y of bar bottoms
  bar_max <- 0.500   # height of ICC=1 bar
  bar_w   <- 0.090
  
  icc_d <- data.frame(
    var  = c("converged", "stance_entropy", "probability", "stance_openness"),
    icc  = c(1.000, 1.000, 0.103, 0.090),
    col  = c("#E74C3C", "#E74C3C", "#27AE60", "#27AE60"),
    xp   = c(0.14, 0.38, 0.63, 0.87),
    neff = c("1,000", "1,000", "3,057", "3,148")
  )
  
  for (i in 1:nrow(icc_d)) {
    bh      <- icc_d$icc[i] * bar_max
    x0      <- icc_d$xp[i]
    bar_top <- bar_b + bh
    
    grid.rect(x = x0, y = bar_b + bh / 2, width = bar_w, height = bh,
              gp = gpar(fill = icc_d$col[i], col = "white", lwd = 0.5))
    
    if (i == 1) {
      # bar1: label to the RIGHT, low vertical (ly = bar_top + 0.07 = 0.71)
      lx <- x0 + 0.096; ly <- bar_top + 0.075
      grid.roundrect(x = lx, y = ly, width = 0.160, height = 0.100,
                     r = unit(3, "pt"),
                     gp = gpar(fill = "#FEF9F9", col = "#E74C3C", lwd = 0.5))
      grid.text("ICC = 1.000",    x = lx, y = ly + 0.022,
                gp = gpar(fontsize = 8.5, col = "#C0392B", fontface = "bold"))
      grid.text(paste0("N_eff = ", icc_d$neff[i]), x = lx, y = ly - 0.022,
                gp = gpar(fontsize = 8, col = "#C0392B"))
      grid.lines(x = c(lx - 0.080, x0 + bar_w / 2),
                 y = c(ly - 0.010, bar_top + 0.010),
                 gp = gpar(col = "#C0392B", lwd = 0.8, lty = "dashed"))
      
    } else if (i == 2) {
      # bar2: label to the RIGHT, higher vertical (clear gap from bar1 label)
      lx <- x0 + 0.096; ly <- bar_top + 0.175  # 0.64+0.175=0.815, stays in vp
      grid.roundrect(x = lx, y = ly, width = 0.160, height = 0.100,
                     r = unit(3, "pt"),
                     gp = gpar(fill = "#FEF9F9", col = "#E74C3C", lwd = 0.5))
      grid.text("ICC = 1.000",    x = lx, y = ly + 0.022,
                gp = gpar(fontsize = 8.5, col = "#C0392B", fontface = "bold"))
      grid.text(paste0("N_eff = ", icc_d$neff[i]), x = lx, y = ly - 0.022,
                gp = gpar(fontsize = 8, col = "#C0392B"))
      grid.lines(x = c(lx - 0.080, x0 + bar_w / 2),
                 y = c(ly - 0.010, bar_top + 0.010),
                 gp = gpar(col = "#C0392B", lwd = 0.8, lty = "dashed"))
      
    } else {
      # ICC~0.1 bars: label directly above
      lab_y <- bar_top + 0.065
      grid.text(paste0("ICC = ", icc_d$icc[i]),
                x = x0, y = lab_y + 0.020,
                gp = gpar(fontsize = 8.5, col = icc_d$col[i], fontface = "bold"))
      grid.text(paste0("N_eff = ", icc_d$neff[i]),
                x = x0, y = lab_y - 0.015,
                gp = gpar(fontsize = 8, col = icc_d$col[i]))
    }
    
    # x-axis variable name
    grid.text(icc_d$var[i], x = x0, y = bar_b - 0.060,
              gp = gpar(fontsize = 9, col = clr$text))
  }
  
  # Threshold line at ICC=0.1: y = bar_b + 0.1*bar_max = 0.14+0.05 = 0.19
  thresh_y <- bar_b + 0.1 * bar_max
  grid.lines(x = c(0.04, 0.60), y = c(thresh_y, thresh_y),
             gp = gpar(col = clr$sub, lwd = 0.8, lty = "dashed"))
  grid.text("ICC = 0.1 threshold", x = 0.310, y = thresh_y + 0.038,
            gp = gpar(fontsize = 7.5, col = clr$sub, fontface = "italic"))
  
  # Legend at very bottom of ICC viewport
  grid.roundrect(x = 0.500, y = 0.045, width = 0.940, height = 0.080,
                 r = unit(4, "pt"),
                 gp = gpar(fill = "#FAFAFA", col = "#BDC3C7", lwd = 0.8))
  grid.rect(x = 0.085, y = 0.058, width = 0.040, height = 0.025,
            gp = gpar(fill = "#E74C3C", col = NA))
  grid.text("Session-shared (ICC \u2248 1.0)  \u2014  agent-level model inappropriate",
            x = 0.435, y = 0.058,
            gp = gpar(fontsize = 8, col = "#922B21"))
  grid.rect(x = 0.085, y = 0.028, width = 0.040, height = 0.025,
            gp = gpar(fill = "#27AE60", col = NA))
  grid.text("Agent-level variation exists  \u2014  both levels appropriate",
            x = 0.418, y = 0.028,
            gp = gpar(fontsize = 8, col = "#1E8449"))
  
  popViewport()  # end ICC viewport
}

png(paste0(OUT, "/figA_crosslevel_architecture_v6.png"),
    width = 4200, height = 3400, res = 300)
figA(); dev.off()
cat("  -> figA_crosslevel_architecture_v6.png\n")


# ============================================================
# FIG B — FIX B2
# Problem: both labels overlap the point/error-bar region
# Fix:
#   Estimate label -> y = ci_hi + 0.010 (above upper CI whisker)
#   CI label       -> y = ci_lo - 0.012 (below lower CI whisker)
#   y-axis limit: top = 0.040, bottom = -0.135
# ============================================================
cat("Generating Fig B...\n")

df_b1 <- data.frame(
  spec = factor(c(
    "Agent-level\n(Y = probability,\nM = stance_direction)",
    "Session-level\n(Y = converged,\nM = mean_openness)"
  ), levels = c(
    "Agent-level\n(Y = probability,\nM = stance_direction)",
    "Session-level\n(Y = converged,\nM = mean_openness)"
  )),
  PM   = c(35.2, 60.3),
  fill = c(clr$agent, clr$sess)
)

pb1_base <- ggplot(df_b1, aes(x = spec, y = PM, fill = spec)) +
  geom_col(width = 0.45, color = "white") +
  geom_text(aes(label = paste0("PM = ", PM, "%")),
            vjust = -0.5, size = 4.6, fontface = "bold", color = clr$text) +
  scale_fill_manual(values = c(clr$agent, clr$sess)) +
  scale_y_continuous(limits = c(0, 82), labels = function(x) paste0(x, "%")) +
  labs(title = "(B1) Proportion mediated by specification",
       subtitle = paste0("Both specs agree in sign & significance.\n",
                         "Magnitude diverges due to cross-level design."),
       x = NULL, y = "Proportion mediated (%)") +
  base_theme +
  theme(axis.text.x  = element_text(size = 9, face = "bold", lineheight = 1.25),
        plot.margin  = margin(6, 8, 12, 8))

pb2 <- {
  df_b2 <- data.frame(
    spec   = factor(c("Agent-level\n(z = \u221215.686)", "Session-level\n(z = \u22124.899)"),
                    levels = c("Agent-level\n(z = \u221215.686)", "Session-level\n(z = \u22124.899)")),
    ind    = c(-0.02888, -0.06213),
    ci_lo  = c(-0.033, -0.088),
    ci_hi  = c(-0.025, -0.039),
    fill   = c(clr$agent, clr$sess)
  )
  ggplot(df_b2, aes(x = spec, y = ind, color = spec)) +
    geom_hline(yintercept = 0, linewidth = 0.8, color = clr$sub) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                  width = 0.12, linewidth = 1.3) +
    geom_point(size = 5.5) +
    # Estimate label: ABOVE upper CI whisker (clear of point and bar)
    geom_text(aes(y = ci_hi + 0.010,
                  label = formatC(ind, digits = 5, format = "f")),
              vjust = 0, size = 3.2, fontface = "bold") +
    # CI label: BELOW lower CI whisker (clear of everything)
    geom_text(aes(y = ci_lo - 0.012,
                  label = paste0("[", ci_lo, ", ", ci_hi, "]")),
              vjust = 1, size = 3.0) +
    scale_color_manual(values = c(clr$agent, clr$sess)) +
    scale_y_continuous(limits = c(-0.135, 0.040)) +
    labs(title = "(B2) Indirect effect with 95% CI",
         subtitle = "Both bootstrap CIs exclude zero; direction identical",
         x = NULL, y = "Indirect effect") +
    base_theme +
    theme(axis.text.x = element_text(size = 9, face = "bold", lineheight = 1.2))
}

pb3 <- {
  df_b3 <- data.frame(
    swap  = factor(c("M-swap\n(\u0394z = 2.02)", "Y-swap\n(\u0394z = 6.76)"),
                   levels = c("M-swap\n(\u0394z = 2.02)", "Y-swap\n(\u0394z = 6.76)")),
    delta = c(2.02, 6.76), fill = c("#BDC3C7", clr$rc)
  )
  ggplot(df_b3, aes(x = swap, y = delta, fill = swap)) +
    geom_col(width = 0.45, color = "white") +
    geom_text(aes(label = paste0("\u0394z = ", delta)),
              vjust = -0.5, size = 4.1, fontface = "bold") +
    scale_fill_manual(values = c("#BDC3C7", clr$rc)) +
    scale_y_continuous(limits = c(0, 10.5)) +
    annotate("text", x = 1.5, y = 9.6,
             label = "Y operationalisation\nis the dominant driver",
             size = 3.3, color = clr$rc, fontface = "italic", hjust = 0.5) +
    labs(title = "(B3) Y\u00d7M 2\u00d72 decomposition",
         subtitle = "Y-swap \u226b M-swap: precision gap driven by ICC structure",
         x = NULL, y = "\u0394z (absolute)") +
    base_theme
}

png(paste0(OUT, "/figB_PM_comparison_v6.png"), width = 4600, height = 2100, res = 300)
grid.newpage()
pushViewport(viewport(layout = grid.layout(
  3, 3,
  widths  = unit(c(1, 1, 1), "null"),
  heights = unit(c(0.055, 0.720, 0.225), "null")
)))

pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1:3))
grid.text("Fig. B  PM Specification Comparison: Why Two PM Estimates Are Both Correct",
          x = 0.5, y = 0.55,
          gp = gpar(fontsize = 13, fontface = "bold", col = clr$text))
popViewport()

pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1))
print(pb1_base, vp = viewport(x = 0.5, y = 0.5, width = 0.95, height = 0.96))
popViewport()

pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 1))
grid.roundrect(x = 0.50, y = 0.52, width = 0.92, height = 0.86,
               r = unit(4, "pt"),
               gp = gpar(fill = "#FAFAFA", col = "#BDC3C7", lwd = 0.8))
grid.text("Agent-level",   x = 0.28, y = 0.92,
          gp = gpar(fontsize = 8.5, fontface = "bold", col = clr$agent))
grid.text("Session-level", x = 0.72, y = 0.92,
          gp = gpar(fontsize = 8.5, fontface = "bold", col = clr$sess))
grid.lines(x = c(0.04, 0.96), y = c(0.83, 0.83),
           gp = gpar(col = "#BDC3C7", lwd = 0.7))
grid.lines(x = c(0.50, 0.50), y = c(0.10, 0.83),
           gp = gpar(col = "#E0E0E0", lwd = 0.6))
grid.text("indirect = \u22120.02888", x = 0.28, y = 0.70,
          gp = gpar(fontsize = 8, col = clr$text))
grid.text("indirect = \u22120.06213", x = 0.72, y = 0.70,
          gp = gpar(fontsize = 8, col = clr$text))
grid.text("c = \u22120.082", x = 0.28, y = 0.55,
          gp = gpar(fontsize = 8, col = clr$text))
grid.text("c = \u22120.103", x = 0.72, y = 0.55,
          gp = gpar(fontsize = 8, col = clr$text))
grid.text("z = \u221215.686", x = 0.28, y = 0.40,
          gp = gpar(fontsize = 8, col = clr$text))
grid.text("z = \u22124.899",  x = 0.72, y = 0.40,
          gp = gpar(fontsize = 8, col = clr$text))
grid.text("95% CI [\u22120.033, \u22120.025]", x = 0.28, y = 0.25,
          gp = gpar(fontsize = 8, col = clr$text))
grid.text("95% CI [\u22120.088, \u22120.039]", x = 0.72, y = 0.25,
          gp = gpar(fontsize = 8, col = clr$text))
popViewport()

pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 2))
print(pb2, vp = viewport(x = 0.5, y = 0.5, width = 0.95, height = 0.96))
popViewport()

pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 3))
print(pb3, vp = viewport(x = 0.5, y = 0.5, width = 0.95, height = 0.96))
popViewport()

pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 2:3))
grid.text(
  paste0("Agent-level: N = 4,000 agent-rounds, cluster = session_id.  ",
         "Session-level: n = 1,000 sessions.  ",
         "Y\u00d7M decomposition: 5,000 bootstrap iterations per cell."),
  x = 0.5, y = 0.82,
  gp = gpar(fontsize = 8, col = clr$sub))
popViewport()

dev.off()
cat("  -> figB_PM_comparison_v6.png\n")

cat("\n=== v6.2 figures saved to", OUT, "===\n")