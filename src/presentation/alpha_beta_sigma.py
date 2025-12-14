from manim import *
import numpy as np

class AlphaBetaSigmaScene(Scene):
    def construct(self):
        # 1. Title Slide
        title = Text("Alpha, Beta, and Sigma", font_size=48, color=BLUE)
        subtitle = Text("Decomposing Stock Returns", font_size=32, color=GREY).next_to(title, DOWN)
        
        self.play(Write(title), FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # 2. Formula Breakdown
        formula = MathTex(
            "R_i", "=", "\\alpha", "+", "\\beta", "R_m", "+", "\\epsilon"
        ).scale(1.5)
        
        # Coloring the parts
        formula[0].set_color(WHITE) # Ri
        formula[2].set_color(GREEN) # Alpha
        formula[4].set_color(YELLOW) # Beta
        formula[5].set_color(BLUE) # Rm
        formula[7].set_color(RED) # Epsilon/Sigma
        
        self.play(Write(formula))
        self.wait(1)
        
        # Explanations
        alpha_text = Text("Alpha: The 'Edge' (Excess Return)", color=GREEN, font_size=24).next_to(formula, UP, buff=1)
        beta_text = Text("Beta: Market Sensitivity", color=YELLOW, font_size=24).next_to(formula, DOWN, buff=1)
        sigma_text = Text("Epsilon: Idiosyncratic Risk (Sigma)", color=RED, font_size=24).next_to(beta_text, DOWN, buff=0.5)
        
        self.play(Indicate(formula[2]), FadeIn(alpha_text))
        self.wait(2)
        self.play(Indicate(formula[4]), Indicate(formula[5]), FadeIn(beta_text))
        self.wait(2)
        self.play(Indicate(formula[7]), FadeIn(sigma_text))
        self.wait(3)
        
        group = VGroup(formula, alpha_text, beta_text, sigma_text)
        self.play(FadeOut(group))
        
        # 3. Visual Analogy (Graph)
        ax = Axes(
            x_range=[-0.05, 0.05, 0.01],
            y_range=[-0.05, 0.05, 0.01],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": False},
        ).add_coordinates()
        
        labels = ax.get_axis_labels(x_label="Market Return (Rm)", y_label="Stock Return (Ri)")
        
        self.play(Create(ax), Write(labels))
        
        # Generate some synthetic data
        np.random.seed(42)
        beta_val = 1.2
        alpha_val = 0.005
        market_returns = np.random.normal(0, 0.015, 50)
        stock_returns = alpha_val + beta_val * market_returns + np.random.normal(0, 0.01, 50)
        
        dots = VGroup()
        for x, y in zip(market_returns, stock_returns):
            dot = Dot(ax.coords_to_point(x, y), color=BLUE_A, radius=0.05)
            dots.add(dot)
            
        self.play(FadeIn(dots, lag_ratio=0.05))
        self.wait(1)
        
        # Draw Regression Line (Beta)
        line = ax.plot(lambda x: alpha_val + beta_val * x, color=YELLOW)
        beta_label = MathTex("\\beta = 1.2").next_to(line, RIGHT, buff=0.2).set_color(YELLOW)
        
        self.play(Create(line), Write(beta_label))
        self.wait(1)
        
        # Highlight Alpha (Intercept)
        intercept_point = ax.coords_to_point(0, alpha_val)
        alpha_dot = Dot(intercept_point, color=GREEN, radius=0.1)
        alpha_arrow = Arrow(start=ax.coords_to_point(0.01, alpha_val), end=intercept_point, color=GREEN)
        alpha_lbl = Text("Alpha > 0", color=GREEN, font_size=20).next_to(alpha_arrow, RIGHT)
        
        self.play(FadeIn(alpha_dot), Create(alpha_arrow), Write(alpha_lbl))
        self.wait(2)
        
        # Highlight Sigma (Residual)
        # Pick one point far from line
        idx = np.argmax(np.abs(stock_returns - (alpha_val + beta_val * market_returns)))
        pt_x, pt_y = market_returns[idx], stock_returns[idx]
        pred_y = alpha_val + beta_val * pt_x
        
        res_line = Line(ax.coords_to_point(pt_x, pred_y), ax.coords_to_point(pt_x, pt_y), color=RED)
        sigma_lbl = Text("Sigma (Risk)", color=RED, font_size=20).next_to(res_line, LEFT)
        
        self.play(Create(res_line), Write(sigma_lbl))
        self.wait(2)
        
        # Conclusion
        self.play(FadeOut(VGroup(ax, labels, dots, line, beta_label, alpha_dot, alpha_arrow, alpha_lbl, res_line, sigma_lbl)))
        
        final_text = Text("Control Beta with Hedging.\nHunt for Alpha.\nManage Sigma with Diversification.", 
                          font_size=36, t2c={"Beta": YELLOW, "Alpha": GREEN, "Sigma": RED})
        self.play(Write(final_text))
        self.wait(3)
