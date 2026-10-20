"""Run: python ray_tracing_3d.py (desktop Python, not a browser)."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image
from imaging import trajectory, render_image

BG = '#101928'
FG = '#e9f1fc'


def setup_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Yu Gothic', 'Meiryo', 'Noto Sans CJK JP', 'IPAexGothic']:
        if name in names:
            matplotlib.rcParams['font.family'] = name
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


class BlackHoleApp:
    def __init__(self, root):
        setup_font()
        self.root = root
        root.title('光の道筋とブラックホールの像 | Python')
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        window_w = min(1380, screen_w - 80)
        window_h = min(900, screen_h - 100)

        root.geometry(f"{window_w}x{window_h}")
        root.minsize(800, 550)
        style=ttk.Style(); style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG, font=('',10))
        style.configure('TButton', padding=7)
        self.events=queue.Queue(); self.busy=False; self.image=None
        self.paths=[]; self.lines=[]; self.dots=[]; self.frame=0; self.playing=False
        top=ttk.Frame(root, padding=(16,10)); top.pack(fill='x')
        ttk.Label(top,text='BLACK HOLE  /  光はどこへ行く？',font=('',20,'bold')).pack(anchor='w')
        ttk.Label(top,text='① 条件を設定 → ② 光線を発射 / 像を生成 → ③ 生成された像をJPG保存').pack(anchor='w',pady=5)
        self.fig=Figure(figsize=(12,6), facecolor=BG)
        self.ax=self.fig.add_axes([.02,.10,.57,.82],projection='3d')
        self.result_ax=self.fig.add_axes([.64,.36,.32,.58])
        self.ref_ax=self.fig.add_axes([.76,.05,.13,.24])
        for ax in (self.result_ax,self.ref_ax): ax.set_facecolor('#060b14'); ax.set_axis_off()
        self.result_ax.set_title('生成された像 — 実行すると表示',color=FG,fontsize=14)
        self.ref_ax.set_title('重力がない場合',color=FG,fontsize=9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        # 操作パネル全体をまとめる
        bottom_panel = ttk.Frame(root)
        bottom_panel.pack(side="bottom", fill="x")
        controls=ttk.Frame(bottom_panel,padding=(16,5)); controls.pack(fill='x')
        for c in range(4): controls.columnconfigure(c,weight=1)
        self.b=self.slider(controls,0,'光線のねらいのずれ b/M',0,12,7)
        self.psi=self.slider(controls,1,'軌道の向き ψ [度]',0,360,30)
        self.fov=self.slider(controls,2,'画像の画角 [度]',20,65,45)
        self.rotation=self.slider(controls,3,'背景の回転 [度]',0,360,0)
        actions=ttk.Frame(bottom_panel,padding=(16,8)); actions.pack(fill='x')
        ttk.Button(actions,text='光線を発射',command=self.launch).pack(side='left',padx=3)
        ttk.Button(actions,text='3本を比較',command=lambda:self.launch(True)).pack(side='left',padx=3)
        ttk.Button(actions,text='停止 / 再開',command=self.toggle).pack(side='left',padx=3)
        ttk.Button(actions,text='視点を戻す',command=self.reset_view).pack(side='left',padx=3)
        self.kind=tk.StringVar(value='色つき格子')
        ttk.Combobox(actions,textvariable=self.kind,values=['色つき格子','星空'],state='readonly',width=10).pack(side='left',padx=8)
        self.res=tk.StringVar(value='256')
        ttk.Combobox(actions,textvariable=self.res,values=['128','256','512'],state='readonly',width=5).pack(side='left')
        self.generate_button=ttk.Button(actions,text='像を生成',command=self.generate); self.generate_button.pack(side='left',padx=8)
        self.save_button=ttk.Button(actions,text='JPGで保存',command=self.save,state='disabled'); self.save_button.pack(side='left',padx=3)
        self.message=tk.StringVar(value='左の図はドラッグで回転できます。まず「3本を比較」を押してください。')
        ttk.Label(bottom_panel,textvariable=self.message,wraplength=1300).pack(fill='x',padx=20,pady=5)
        ttk.Label(bottom_panel,text='黒：事象の地平面 r=2M  /  点線：光子球 r=3M  /  奥の面：観測像の展示スクリーン（光の到達面ではありません）\n光線は経路を見せるための速さで動きます。自転なし・G=c=M=1。bは選択した1本、画角は画像全体の設定です。',wraplength=1300).pack(fill='x',padx=20,pady=(0,10))
        # 操作パネルを確保した後、残りの領域にグラフを配置
        self.canvas.get_tk_widget().pack(
            side="top",
            fill="both",
            expand=True
        )
        self.draw_scene(); self.canvas.draw()
        root.after(40,self.tick)
        self.canvas.mpl_connect('button_press_event',self.pick)
        root.protocol('WM_DELETE_WINDOW',root.destroy)

    def slider(self,parent,col,label,lo,hi,value):
        panel=ttk.Frame(parent,padding=6); panel.grid(row=0,column=col,sticky='ew')
        var=tk.DoubleVar(value=value); text=tk.StringVar()
        def update(*_): text.set(f'{label} : {var.get():.2f}')
        var.trace_add('write',update); update()
        ttk.Label(panel,textvariable=text).pack(anchor='w')
        ttk.Scale(panel,from_=lo,to=hi,variable=var).pack(fill='x')
        return var

    def draw_scene(self):
        ax=self.ax; ax.clear(); ax.set_facecolor(BG)
        ax.computed_zorder=False
        u,v=np.meshgrid(np.linspace(0,2*np.pi,48),np.linspace(0,np.pi,24))
        xyz=(np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v))
        ax.plot_surface(*(2*a for a in xyz),color='#030509',shade=True,zorder=2)
        ax.plot_wireframe(*(3*a for a in xyz),rstride=5,cstride=6,color='#6a829c',alpha=.25,linewidth=.5,zorder=3)
        yy,zz=np.meshgrid(np.linspace(-12,12,40),np.linspace(-12,12,40))
        if self.image is None:
            colors=np.zeros((40,40,4)); colors[:]=[.12,.24,.36,.7]
        else:
            colors=np.asarray(Image.fromarray(self.image).resize((40,40)))/255.
            # Array rows top-to-bottom, screen z bottom-to-top.
            colors=colors[::-1]
        ax.plot_surface(np.full_like(yy,-19),yy,zz,facecolors=colors,shade=False,rcount=40,ccount=40,zorder=0)
        ax.text(-19,0,14,'観測像の展示スクリーン',color=FG,ha='center',fontsize=9)
        ax.scatter([30],[0],[0],color='#ffffff',s=35)
        ax.text(29,0,3,'観測位置',color=FG,fontsize=9)
        ax.set(xlim=(-22,32),ylim=(-16,16),zlim=(-16,16))
        ax.set_box_aspect((54,32,32)); ax.view_init(22,42)
        ax.set_axis_off(); ax.set_title('3D：光の経路をたどる',color=FG,fontsize=14)
        self.lines=[]; self.dots=[]
        for path,status,b in self.paths:
            color='#ffad66' if status=='捕獲' else '#66e4e0'
            line,=ax.plot([],[],[],color=color,lw=1.8,zorder=4,label=f'b={b:.3f}  {status}')
            dot,=ax.plot([],[],[],'o',color='white',ms=4,zorder=5)
            self.lines.append(line); self.dots.append(dot)
        if self.lines: ax.legend(loc='lower left',facecolor=BG,labelcolor=FG,fontsize=9)

    def launch(self,compare=False):
        values=[4.5,5.2,7.] if compare else [self.b.get()]
        psi=np.deg2rad(self.psi.get())
        self.paths=[]
        for b in values:
            path,status=trajectory(b,psi)
            self.paths.append((path,status,b))
        self.frame=0; self.playing=True; self.draw_scene()
        self.message.set(' / '.join(f'b={b:.3f}：{s}' for _,s,b in self.paths)+'。水色は脱出、オレンジは捕獲。')
        self.canvas.draw_idle()

    def toggle(self): self.playing=not self.playing
    def reset_view(self): self.ax.view_init(22,42); self.canvas.draw_idle()

    def generate(self):
        if self.busy: return
        self.busy=True; self.generate_button.configure(state='disabled')
        args=(int(self.res.get()),self.fov.get(),np.deg2rad(self.rotation.get()),self.kind.get()=='色つき格子')
        self.message.set('光線を数値計算しています。3D表示はそのまま操作できます。')
        def worker():
            try:
                result=render_image(*args,progress=lambda p:self.events.put(('progress',p)))
                self.events.put(('done',result))
            except Exception as e: self.events.put(('error',str(e)))
        threading.Thread(target=worker,daemon=True).start()

    def tick(self):
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=='progress': self.message.set(f'像を生成中… {data:.0%}')
                else:
                    self.busy=False; self.generate_button.configure(state='normal')
                    if kind=='error': self.message.set('計算エラー：'+data); continue
                    self.image,reference,self.meta=data
                    self.result_ax.clear(); self.result_ax.imshow(self.image); self.result_ax.set_axis_off()
                    self.result_ax.set_title(f'生成された像  {self.meta["n"]} × {self.meta["n"]}',color=FG,fontsize=14)
                    self.ref_ax.clear(); self.ref_ax.imshow(reference); self.ref_ax.set_axis_off()
                    self.ref_ax.set_title('重力がない場合',color=FG,fontsize=9)
                    self.draw_scene(); self.frame=0; self.playing=bool(self.paths)
                    self.save_button.configure(state='normal')
                    self.message.set(f'{self.meta["rays"]:,}画素を追跡しました。黒は捕獲。像をクリックすると、その方向の光線を表示します。未判定：{self.meta["unresolved"]}')
                    self.canvas.draw_idle()
        except queue.Empty: pass
        if self.playing and self.paths:
            self.frame+=1
            for (path,_,_),line,dot in zip(self.paths,self.lines,self.dots):
                end=min(len(path),max(2,int(len(path)*self.frame/180)))
                shown=path[:end:max(1,end//650)]
                line.set_data_3d(*shown.T); dot.set_data_3d(*path[end-1:end].T)
            if self.frame>=180: self.playing=False
            self.canvas.draw_idle()
        self.root.after(40,self.tick)

    def pick(self,event):
        if self.image is None or event.inaxes is not self.result_ax or event.xdata is None: return
        n=self.meta['n']; t=np.tan(np.deg2rad(self.meta['fov']/2))
        y=(event.xdata+.5-n/2)/(n/2)*t; z=-(event.ydata+.5-n/2)/(n/2)*t
        rho=np.hypot(y,z); b=30*rho/np.sqrt(1+rho*rho)/np.sqrt(1-2/30)
        self.b.set(b); self.psi.set(np.rad2deg(np.arctan2(z,y))%360); self.launch()

    def save(self):
        if self.image is None: return
        path=filedialog.asksaveasfilename(parent=self.root,title='生成された像をJPGで保存',defaultextension='.jpg',initialfile='blackhole_image.jpg',filetypes=[('JPEG','*.jpg'),('JPEG','*.jpeg')])
        if not path: return
        try:
            Image.fromarray(self.image).save(path,format='JPEG',quality=95,subsampling=0)
            self.message.set('保存しました：'+path)
        except OSError as e: messagebox.showerror('保存できませんでした',str(e))


if __name__=='__main__':
    app_root=tk.Tk()
    app=BlackHoleApp(app_root)
    app_root.mainloop()
