"""Independent, fixed-template cart measurement. No self-updating drift."""
import cv2
import numpy as np


class CartTemplate:
    def __init__(self):
        self.template=None
        self.offset=None
        self.reference_y=None
        self.shape=None
        self.invalid_reason=None

    def select(self, frame, rectangle, pivot):
        x,y,w,h=map(int,rectangle)
        height,width=frame.shape[:2]
        if x<0 or y<0 or w<12 or h<12 or x+w>width or y+h>height:
            raise ValueError('Select a cart rectangle at least 12 pixels wide and high, inside the image')
        if w>width*.4 or h>height*.4:
            raise ValueError('Select a small textured part of the rigid cart, not the background')
        patch=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
        if patch.std()<8:
            raise ValueError('Cart patch has too little texture')
        self.invalid_reason=None
        self.template=patch.copy()
        self.offset=np.asarray(pivot,float)-[x,y]
        self.reference_y=y
        self.shape=frame.shape[:2]

    def detect(self, frame):
        if self.template is None:
            return None,None,'not selected'
        if self.invalid_reason is not None:
            return None,None,self.invalid_reason
        if frame.shape[:2]!=self.shape:
            return None,None,'camera dimensions changed; reselect cart'
        h,w=self.template.shape
        margin=max(40,round(frame.shape[0]*.15))
        y0=max(0,self.reference_y-margin)
        y1=min(frame.shape[0],self.reference_y+h+margin)
        gray=cv2.cvtColor(frame[y0:y1],cv2.COLOR_BGR2GRAY)
        scores=cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED)
        _,best,_,(x,y)=cv2.minMaxLoc(scores)
        excluded=scores.copy()
        dx,dy=max(6,w//3),max(6,h//3)
        excluded[max(0,y-dy):y+dy+1,max(0,x-dx):x+dx+1]=-1
        second=float(excluded.max())
        if best<.80 or best-second<.06:
            return None,float(best),'cart appearance weak or ambiguous'
        def refine(a,b,c):
            denominator=a-2*b+c
            return float(np.clip(.5*(a-c)/denominator,-.5,.5)) if abs(denominator)>1e-6 else 0.
        sx=refine(scores[y,x-1],scores[y,x],scores[y,x+1]) if 0<x<scores.shape[1]-1 else 0.
        sy=refine(scores[y-1,x],scores[y,x],scores[y+1,x]) if 0<y<scores.shape[0]-1 else 0.
        p=np.array([x+sx,y+y0+sy])+self.offset
        return tuple(p),float(best),'measured'
