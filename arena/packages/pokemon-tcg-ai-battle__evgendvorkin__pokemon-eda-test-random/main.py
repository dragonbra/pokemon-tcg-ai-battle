import os, random, math, time
from collections import defaultdict
from cg.api import (
    AreaType, CardType, EnergyType, Observation, OptionType,
    Pokemon, SelectContext, all_card_data, to_observation_class
)

_SEARCH_OK = False
try:
    from cg.api import search_begin, search_step, search_release
    _SEARCH_OK = True
except:
    pass

USE_SEARCH = True
SEARCH_TIME_BUDGET = 1.5
SEARCH_MAX_CANDIDATES = 6

C_MAKUHITA=673; C_HARIYAMA=674; C_LUNATONE=675; C_SOLROCK=676; C_RIOLU=677; C_MEGA_LUCARIO_EX=678
C_FIGHTING_ENERGY=6; C_DUSK_BALL=1102; C_SWITCH=1123; C_PREMIUM_POWER_PRO=1141
C_FIGHTING_GONG=1142; C_POKE_PAD=1152; C_HERO_CAPE=1159; C_BOSS_ORDERS=1182
C_CARMINE=1192; C_LILLIE=1227; C_GRAVITY_MOUNTAIN=1252
C_LILLIES_PEARL=1172; C_LEGACY_ENERGY=12
MEGA_BRAVE=983; LOW_DECK=8

DECK=[673,673,674,674,675,675,676,676,676,677,677,677,678,678,678,678,1102,1102,1102,1102,1123,1123,1141,1141,1141,1141,1142,1142,1142,1142,1152,1152,1152,1152,1159,1182,1182,1192,1192,1192,1192,1227,1227,1227,1227,1252,1252,6,6,6,6,6,6,6,6,6,6,6,6,6]

all_card=all_card_data()
card_table={c.cardId:c for c in all_card}

plan=None; pre_turn=-1; ability_used=False

class AttackPlan:
    def __init__(self,attacker=-1,target=-1,attack_index=-1,remain_hp=-1,needs_energy=False):
        self.attacker=attacker; self.target=target; self.attack_index=attack_index
        self.remain_hp=remain_hp; self.needs_energy=needs_energy

def safe_get(seq,index):
    try:
        if seq is None or index is None or index<0 or index>=len(seq): return None
        return seq[index]
    except: return None

def get_card(obs,area,index,player_index):
    try:
        p=obs.current.players[player_index]
        if area==AreaType.HAND: return safe_get(p.hand,index)
        if area==AreaType.DISCARD: return safe_get(p.discard,index)
        if area==AreaType.ACTIVE: return safe_get(p.active,index)
        if area==AreaType.BENCH: return safe_get(p.bench,index)
        if area==AreaType.PRIZE: return safe_get(p.prize,index)
        if area==AreaType.STADIUM: return safe_get(obs.current.stadium,index)
        if area==AreaType.DECK: return safe_get(obs.select.deck,index)
        return None
    except: return None

def prize_count(pokemon):
    data=card_table.get(pokemon.id)
    if data is None: return 1
    cnt=3 if data.megaEx else 2 if data.ex else 1
    for c in pokemon.energyCards:
        if c.id==C_LEGACY_ENERGY: cnt-=1
    for c in pokemon.tools:
        if c.id==C_LILLIES_PEARL: cnt-=1
    return max(0,cnt)

def target_score(pokemon):
    data=card_table.get(pokemon.id)
    if data is None: return prize_count(pokemon)*1000+pokemon.hp
    score=prize_count(pokemon)*1000+len(pokemon.energies)*150+len(pokemon.tools)*100
    if data.stage2: score+=250
    elif data.stage1: score+=130
    score+=pokemon.hp
    return score

def legal_fallback(select):
    try:
        n=len(select.option); k=min(max(0,select.minCount),n); return list(range(k))
    except: return []

def normalize_selection(ranked,scores,select):
    n=len(select.option); minc=max(0,min(select.minCount,n)); maxc=max(minc,min(select.maxCount,n))
    out,seen=[],set()
    for i in ranked:
        if not(0<=i<n) or i in seen: continue
        score=scores[i] if i<len(scores) else 0
        if score>0 or len(out)<minc: out.append(i); seen.add(i)
        if len(out)>=maxc: break
    for i in range(n):
        if len(out)>=minc: break
        if i not in seen: out.append(i); seen.add(i)
    return out

class LucarioPolicy:
    def __init__(self,obs):
        self.obs=obs; self.state=obs.current; self.select=obs.select; self.context=self.select.context
        self.my_index=self.state.yourIndex; self.op_index=1-self.my_index
        self.me=self.state.players[self.my_index]; self.op=self.state.players[self.op_index]
        self.field=defaultdict(int); self.hand=defaultdict(int); self.discard=defaultdict(int)
        self.can_switch=False; self.can_gust=False; self.can_attack=False; self.can_mega_brave=False
        self.stadium_id=self.state.stadium[0].id if self.state.stadium else 0
        self._count(); self._scan()
    
    def _count(self):
        for poke in self.me.active+self.me.bench:
            if poke is not None: self.field[poke.id]+=1
        for card in self.me.hand: self.hand[card.id]+=1
        for card in self.me.discard: self.discard[card.id]+=1
    
    def _scan(self):
        if self.context!=SelectContext.MAIN: return
        for opt in self.select.option:
            if opt.type==OptionType.PLAY:
                card=get_card(self.obs,AreaType.HAND,opt.index,self.my_index)
                if card and card.id==C_SWITCH: self.can_switch=True
                elif card and card.id==C_BOSS_ORDERS: self.can_gust=True
            elif opt.type==OptionType.RETREAT: self.can_switch=True
            elif opt.type==OptionType.ATTACK:
                self.can_attack=True
                if opt.attackId==MEGA_BRAVE: self.can_mega_brave=True
    
    def _my_board(self): return self.me.active+self.me.bench
    def _op_board(self): return self.op.active+self.op.bench
    def _low_deck(self): return self.me.deckCount<=LOW_DECK
    def _lucario_line(self): return self.field[C_RIOLU]+self.field[C_MEGA_LUCARIO_EX]
    
    def _active_pokemon(self):
        return self.me.active[0] if self.me.active and self.me.active[0] is not None else None
    
    def _op_has_big_stage1(self):
        """Есть ли у соперника Stage 1 ex/Mega которого Gravity Mountain добивает?"""
        for p in self._op_board():
            if p is None: continue
            data=card_table.get(p.id)
            if data and data.stage1 and (data.ex or data.megaEx) and data.hp>=270:
                return True
        return False
    
    def _base_attack(self,poke,atk_idx):
        if poke.id==C_MEGA_LUCARIO_EX:
            if atk_idx==0: return(1,130,60*min(3,self.discard[C_FIGHTING_ENERGY]))
            else: return(2,270,0)
        elif poke.id==C_HARIYAMA and atk_idx==0: return(3,210,0)
        elif poke.id==C_SOLROCK and atk_idx==0 and self.field[C_LUNATONE]>=1: return(1,70,0)
        return None
    
    def _plan_attack(self):
        global plan; plan=AttackPlan()
        if self.state.turn<2: return
        best_score=-1
        for ai,poke in enumerate(self._my_board()):
            if poke is None: continue
            if ai!=0 and not self.can_switch: break
            for atk_idx in range(2):
                atk=self._base_attack(poke,atk_idx)
                if atk is None: continue
                need_e,base_dmg,base_scr=atk
                e_cnt=len(poke.energies)
                if atk_idx==1 and ai==0 and e_cnt>=2 and not self.can_mega_brave: break
                needs_e=False
                if e_cnt<need_e:
                    if self.hand[C_FIGHTING_ENERGY]>=1 and not self.state.energyAttached:
                        e_cnt+=1; needs_e=e_cnt>=need_e
                    if not needs_e: continue
                for ti,tgt in enumerate(self._op_board()):
                    if tgt is None: continue
                    if ti!=0 and not self.can_gust: break
                    dmg=base_dmg
                    tgt_data=card_table.get(tgt.id)
                    if tgt_data:
                        if tgt_data.weakness==EnergyType.FIGHTING: dmg*=2
                        elif tgt_data.resistance==EnergyType.FIGHTING: dmg-=30
                    score=target_score(tgt)
                    prz=prize_count(tgt) if tgt.hp<=dmg else 0
                    if prz==0: score*=dmg/max(1,tgt.hp)
                    if len(self.op.prize)<=prz: score=50000
                    score+=base_scr+220 if ai==0 else 0
                    score+=300 if ti==0 else 0
                    if score>best_score: best_score=score; plan=AttackPlan(ai,ti,atk_idx,tgt.hp-dmg,needs_e)
    
    def _energy_score(self,poke,is_active):
        e=len(poke.energies); s=8000+(10 if is_active else 0)
        if poke.id in{C_RIOLU,C_MEGA_LUCARIO_EX}: s+=100 if e<2 else 0
        elif poke.id in{C_MAKUHITA,C_HARIYAMA}: s+=100 if e<3 else 0
        elif poke.id==C_SOLROCK: s+=20 if e<1 else -100
        elif poke.id==C_LUNATONE: s-=100
        return s
    
    def _score_option(self,opt):
        t=opt.type
        if t==OptionType.NUMBER: return opt.number
        if t==OptionType.YES: return 100 if self.context==SelectContext.IS_FIRST else 1
        if t==OptionType.NO: return 0
        if t==OptionType.CARD: return self._score_card(opt)
        if t==OptionType.PLAY: return self._score_play(opt)
        if t==OptionType.ATTACH: return self._score_attach(opt)
        if t==OptionType.EVOLVE:
            poke=get_card(self.obs,opt.inPlayArea,opt.inPlayIndex,self.my_index)
            if isinstance(poke,Pokemon) and poke.id==C_MAKUHITA and plan.target==0: return -1
            return 9000+(len(poke.energies) if isinstance(poke,Pokemon) else 0)
        if t==OptionType.RETREAT:
            active=self._active_pokemon()
            if active and active.id==C_MEGA_LUCARIO_EX and len(active.energies)>=1:
                return -1
            if active and active.id==C_HARIYAMA and len(active.energies)>=1:
                return -1
            return 2000 if plan.attacker>=1 else -1
        if t==OptionType.ATTACK: return 1100 if(opt.attackId==MEGA_BRAVE)==(plan.attack_index==1) else 1000
        if t==OptionType.ABILITY:
            card=get_card(self.obs,opt.area,opt.index,self.my_index)
            if card and card.id==C_LUNATONE and self._low_deck(): return -1
            return 30000
        return 0
    
    def _score_card(self,opt):
        card=get_card(self.obs,opt.area,opt.index,opt.playerIndex)
        if card is None: return 0
        ctx=self.context
        if ctx==SelectContext.SETUP_ACTIVE_POKEMON:
            if card.id==C_SOLROCK: return 2 if self.state.firstPlayer==self.my_index else 4
            if card.id==C_RIOLU: return 3
            if card.id==C_MAKUHITA: return 1
            return 0
        if ctx in{SelectContext.SWITCH,SelectContext.TO_ACTIVE} and isinstance(card,Pokemon):
            if opt.playerIndex!=self.my_index: return 100 if opt.index==plan.target-1 else 0
            s=len(card.energies)*2
            if card.id==C_MEGA_LUCARIO_EX: s+=20
            elif card.id==C_HARIYAMA: s+=15
            elif card.id==C_RIOLU: s+=4
            return s
        if ctx==SelectContext.TO_HAND:
            s=200-self.hand[card.id]*100
            if card.id==C_RIOLU:
                lc=self._lucario_line()
                s+=-150 if lc>=2 else 40
            elif card.id==C_MEGA_LUCARIO_EX: s+=40 if self.field[C_RIOLU]>=1 else -15
            return s
        if ctx==SelectContext.DISCARD:
            if card.id==C_FIGHTING_ENERGY:
                s=45 if self.hand[card.id]>=2 else 5
                if plan.needs_energy and not self.state.energyAttached: s-=200
                return s
            cnt_hand=self.hand[card.id]
            cnt_field=self.field[card.id]
            cnt_disc=self.discard[card.id]
            if cnt_hand==1 and cnt_field==0:
                if card.id in{C_RIOLU,C_MAKUHITA,C_MEGA_LUCARIO_EX}: return -100
                if card.id==C_BOSS_ORDERS and cnt_disc==0: return -60
                if card.id==C_HERO_CAPE: return -30
            if cnt_hand>=2: return 70
            return 0
        if ctx in{SelectContext.DAMAGE_COUNTER,SelectContext.DAMAGE_COUNTER_ANY} and isinstance(card,Pokemon):
            if opt.playerIndex==self.op_index: return 10000+prize_count(card)*1000-card.hp
            return -target_score(card)
        return 0
    
    def _score_play(self,opt):
        card=get_card(self.obs,AreaType.HAND,opt.index,self.my_index)
        if card is None: return 0
        # POKEMON
        if card.id in{C_LUNATONE,C_SOLROCK} and self.field[card.id]>=1: return -1
        if card.id==C_RIOLU and self._lucario_line()>=2: return -1
        # LUNATONE: не играем если deck почти пуст
        if card.id==C_LUNATONE and self._low_deck(): return -1
        if card.id in{C_MAKUHITA,C_HARIYAMA,C_RIOLU,C_MEGA_LUCARIO_EX,C_LUNATONE,C_SOLROCK}: return 20000
        # SWITCH
        if card.id==C_SWITCH:
            active=self._active_pokemon()
            if active and active.id==C_MEGA_LUCARIO_EX and len(active.energies)>=1:
                return -1
            return 6000 if plan.attacker>0 else -1
        # PREMIUM POWER PRO
        if card.id==C_PREMIUM_POWER_PRO:
            if self.state.supporterPlayed: return -1
            if plan.remain_hp <= 0: return -1
            return 5000 if self.can_attack else 3050
        # BOSS
        if card.id==C_BOSS_ORDERS: return 3200 if plan.target>=1 else -1
        # CARMINE
        if card.id==C_CARMINE:
            if self._low_deck(): return -1
            if self.op.handCount <= 2: return -1
            return 3000
        # LILLIE
        if card.id==C_LILLIE:
            if self._low_deck(): return -1
            if self.op.handCount <= 2: return -1
            return 3100
        # GRAVITY MOUNTAIN — улучшен
        if card.id==C_GRAVITY_MOUNTAIN:
            # Против Stage 2
            if any(p is not None and card_table.get(p.id) and card_table[p.id].stage2 for p in self._op_board()):
                return 3500
            # Против Stage 1 ex/Mega с HP>=270 (Archaludon ex и др.)
            if self._op_has_big_stage1():
                return 3200
            # Если стадиум уже есть — не играем
            if self.stadium_id:
                return -1
            return 1200
        return 10000
    
    def _score_attach(self,opt):
        card=get_card(self.obs,AreaType.HAND,opt.index,self.my_index)
        poke=get_card(self.obs,opt.inPlayArea,opt.inPlayIndex,self.my_index)
        if card is None or not isinstance(poke,Pokemon): return 0
        if card.id==C_HERO_CAPE: return 7000+(200 if poke.id==C_MEGA_LUCARIO_EX else 100 if poke.id==C_RIOLU else 0)
        s=self._energy_score(poke,opt.inPlayArea==AreaType.ACTIVE)
        bi=opt.inPlayIndex if opt.inPlayArea==AreaType.ACTIVE else opt.inPlayIndex+1
        if bi==plan.attacker and plan.needs_energy: s+=200
        return s
    
    def rank(self):
        if not self.select.option or self.select.maxCount==0: return [],[]
        if self.context==SelectContext.MAIN: self._plan_attack()
        scores=[self._score_option(o) for o in self.select.option]
        ranked=[i for i,_ in sorted(enumerate(scores),key=lambda x:x[1],reverse=True)]
        return ranked,scores
    
    def choose(self):
        global ability_used
        ranked,scores=self.rank()
        sel=normalize_selection(ranked,scores,self.select)
        if self.context==SelectContext.MAIN:
            for idx in sel:
                if 0<=idx<len(self.select.option):
                    opt=self.select.option[idx]
                    if opt.type==OptionType.ABILITY:
                        card=get_card(self.obs,opt.area,opt.index,self.my_index)
                        if card and card.id==C_LUNATONE: ability_used=True
        return sel

def evaluate_state(obs):
    st=obs.current
    if st is None: return 0.0
    me=st.players[st.yourIndex]; op=st.players[1-st.yourIndex]
    val=(len(op.prize)-len(me.prize))*10000.0
    for p in ([me.active[0]] if me.active else [])+list(me.bench):
        if p is None: continue
        val+=len(p.energies)*120.0
        if p.id==C_MEGA_LUCARIO_EX: val+=400.0
        if p.id==C_HARIYAMA: val+=200.0
    if me.active and me.active[0] is not None: val+=me.active[0].hp*1.0
    if op.active and op.active[0] is not None: val-=op.active[0].hp*1.5
    val+=me.handCount*5.0
    return val

def search_plan(obs_dict,obs):
    if not(_SEARCH_OK and USE_SEARCH): return None
    sel=obs.select
    if sel is None or sel.context!=SelectContext.MAIN: return None
    t0=time.time()
    sbi=getattr(obs,"search_begin_input",None) or obs_dict.get("search_begin_input")
    if sbi is None: return None
    base=LucarioPolicy(obs).choose()
    candidates=base[:SEARCH_MAX_CANDIDATES]
    best_idx,best_val=None,float("-inf")
    for first in candidates:
        if time.time()-t0>SEARCH_TIME_BUDGET: break
        sid=None
        try:
            res=search_begin(sbi)
            if getattr(res,"error",0)!=0 or res.state is None: return None
            sid=res.state.searchId; cur=res.state.observation
            sel_list=[first]; steps=0
            while steps<40:
                sr=search_step(sid,sel_list)
                if getattr(sr,"error",0)!=0 or sr.state is None: break
                cur=sr.state.observation
                if cur.select is None or cur.current is None: break
                if cur.current.result is not None and cur.current.result!=-1: break
                if cur.current.yourIndex!=obs.current.yourIndex: break
                if cur.select.context!=SelectContext.MAIN:
                    sub=LucarioPolicy(cur).choose()
                    sel_list=sub[:max(1,cur.select.minCount)]
                    steps+=1; continue
                nxt=LucarioPolicy(cur).choose()
                sel_list=[nxt[0]]; steps+=1
                if cur.select.option[nxt[0]].type==OptionType.END:
                    sr2=search_step(sid,sel_list)
                    if sr2.state is not None: cur=sr2.state.observation
                    break
            val=evaluate_state(cur)
            if val>best_val: best_val,best_idx=val,first
        except:
            return None
        finally:
            try:
                if sid is not None: search_release(sid)
            except: pass
    if best_idx is None: return None
    rest=[i for i in base if i!=best_idx]
    return [best_idx]+rest

def agent(obs_dict):
    global pre_turn,ability_used,plan
    try:
        obs=to_observation_class(obs_dict)
    except:
        return DECK if isinstance(obs_dict,dict) and obs_dict.get("select") is None else [0]
    if obs.select is None: return DECK
    if obs.current is not None and pre_turn!=obs.current.turn:
        pre_turn=obs.current.turn; ability_used=False; plan=AttackPlan()
    try:
        ordered=None
        if USE_SEARCH: ordered=search_plan(obs_dict,obs)
        if ordered is None: ordered=LucarioPolicy(obs).choose()
        n=len(obs.select.option)
        ordered=[i for i in ordered if 0<=i<n]
        if not ordered: return legal_fallback(obs.select)
        k=min(obs.select.maxCount,n)
        k=max(k,min(max(1,obs.select.minCount),n))
        return ordered[:k]
    except:
        return legal_fallback(obs.select)
