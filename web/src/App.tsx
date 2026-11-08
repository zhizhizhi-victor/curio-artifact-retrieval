import { useEffect, useMemo, useState } from 'react'
import {
  ArrowUpRight, BookOpen, ChevronDown, ChevronRight, Clock3, Database,
  ExternalLink, Filter, Globe2, Info, Layers3, Menu, Search, SlidersHorizontal,
  Sparkles, X, Zap
} from 'lucide-react'

type Variant = 'baseline' | 'translated' | 'hybrid' | 'adapted' | 'lexical' | 'visual' | 'image_only' | 'subquery_and' | 'compositional'
type SearchMode = 'text' | 'reference' | 'image'
type VisualMethod = 'siglip' | 'dinov2' | 'dinov2_patch' | 'shape' | 'color' | 'texture' | 'siglip_shape' | 'siglip_color' | 'siglip_texture' | 'ensemble'
type Reference = { reference_id: string; title?: string; source?: string; source_url?: string; license?: string; credit_line?: string; date?: string; period?: string; reign?: string }
type Result = {
  item_id: number | string; rank: number; score: number; title?: string; thumb_url?: string; image_url?: string;
  source?: string; source_url?: string; matched_slots?: Array<{slot: string; value?: string; basis?: string}>; evidence?: Array<{kind?: string; text?: string; slot?: string; value?: string; probability?: number}>;
  score_breakdown?: Record<string, number | null | number[]>
}
type SearchResponse = {
  request_id?: string; query: string; parsed?: Record<string, unknown>; results?: Result[];
  timings?: Record<string, number>; model_version?: string; index_version?: string; degraded?: boolean; warnings?: string[]; effective_variant?: string;
  references?: Reference[]; plan?: Record<string, unknown>; retrieval_mode?: string; scoring?: string;
  patch_diagnostics?: {recall_count?: number; accepted_candidates?: number; accepted_references?: number; total_references?: number; status?: 'reranked' | 'fallback_cls' | string};
  confidence?: {level?: string; label?: string; basis?: string; degraded?: boolean; scope?: string; dense_top_item_id?: string; final_top_item_id?: string; signal_agreement?: {status?: string; compared?: number; agreeing?: number} | string; signal_availability?: {dense?: boolean; probes?: boolean; probes_applied?: boolean}; visual_coverage?: {covered?: number; total?: number; ratio?: number | null; status?: string}; ranking_separation?: {top1_top2_margin?: number | null; scope?: string; dense_top_item_id?: string; final_top_item_id?: string}}
}
type FacetEntry = { value: string; count: number }; type Facets = { object_type?: FacetEntry[]; material?: FacetEntry[]; source?: FacetEntry[]; era?: [number, number] }; type Health = { status?: string; items?: number; indexed_images?: number; index_version?: string; expanded_catalogue_count?: number }

type Filters = { object_type: string[]; material: string[]; era: [number, number]; source: string[] }

const variants: { id: Variant; label: string; note: string }[] = [
  { id: 'compositional', label: '多条件组合视觉（推荐）', note: '纯图像：跨语言形态桥接 + 子句软合取（Soft-AND）+ 可靠度校准视觉探针门控' },
  { id: 'subquery_and', label: '子句软合取视觉', note: '纯图像零样本：多条件子句拆解 + Soft-AND + 反向视觉对比抑制（无探针）' },
  { id: 'image_only', label: '图像模型检索（v2）', note: '按图文嵌入相似度排序；生产配置属性重排权重为 0' },
  { id: 'hybrid', label: '元数据辅助', note: '稠密 + 词法 + 槽位融合' },
  { id: 'baseline', label: '基线', note: '单路图文相似度' },
  { id: 'translated', label: '词典扩展', note: '规则词典扩展查询' },
  { id: 'adapted', label: '领域适配', note: '弱监督适配模型' },
  { id: 'lexical', label: '词法', note: 'BM25 / 元数据词法' },
  { id: 'visual', label: '视觉属性重排', note: '混合检索 + 冻结图像特征线性分类器' },
]
const visualMethods: { id: VisualMethod; label: string }[] = [
  { id: 'siglip', label: 'SigLIP 图像嵌入' }, { id: 'dinov2', label: 'DINOv2 图像嵌入' },
  { id: 'dinov2_patch', label: 'DINO 局部特征 + 软轮廓（研究）' },
  { id: 'shape', label: '轮廓形状' }, { id: 'color', label: '前景颜色' }, { id: 'texture', label: '前景纹理统计' },
  { id: 'siglip_shape', label: 'SigLIP + 形状' }, { id: 'siglip_color', label: 'SigLIP + 颜色' },
  { id: 'siglip_texture', label: 'SigLIP + 前景纹理统计' }, { id: 'ensemble', label: '多特征融合（研究）' },
]
const examples = ['明代青花瓷瓶，上面有龙纹', '青花瓷碗', '没有盖的罐子', '莲花纹的碗']
const referenceExamples = ['康熙梅瓶', '明代葫芦瓶', '万历青花盘']
const slotNames: Record<string, string> = { object_type: '器型', material: '材质', motif: '纹饰', motifs: '纹饰', era: '年代', color: '颜色', count: '数量', negations: '排除' }

function formatScore(score?: number) { return typeof score === 'number' ? score.toFixed(4) : '—' }
function formatTime(ms?: number) { return typeof ms === 'number' ? `${Math.round(ms)} ms` : '—' }
function readableSlot(key: string) { return slotNames[key] || key.replaceAll('_', ' ') }
function evidenceText(item: {kind?: string; text?: string; value?: string; probability?: number}) { const label = item.kind === 'image_similarity' ? '图像间相似度' : item.kind === 'similarity' ? '模型相似度' : item.kind === 'visual_probe' ? '视觉分类预测' : '馆藏元数据'; const hasProbabilityText = typeof item.text === 'string' && item.text.includes('概率'); const probability = item.kind === 'visual_probe' && !hasProbabilityText && typeof item.probability === 'number' && Number.isFinite(item.probability) ? `（属性预测概率 ${(item.probability * 100).toFixed(1)}%）` : ''; return `[${label}] ${item.text || ''}${probability}` }
function confidenceText(value?: SearchResponse['confidence']) {
  if (!value) return '后端未提供可信性摘要。'
  const coverage = value.visual_coverage
  const coverageText = coverage?.total ? `视觉属性覆盖 ${coverage.covered || 0}/${coverage.total}` : '没有可量化的视觉属性覆盖'
  const margin = value.ranking_separation?.top1_top2_margin
  const scope = value.scope || value.ranking_separation?.scope
  const marginLabel = scope === 'dinov2_cls_recall_before_patch_rerank' ? '局部特征重排前 DINO CLS 召回首两名余弦差值' : scope === 'reference_image_before_fusion' ? '融合前参考图嵌入首两名余弦差值' : '融合前嵌入首两名余弦差值'
  const marginText = typeof margin === 'number' && Number.isFinite(margin) ? `${marginLabel} ${margin.toFixed(6)}` : '融合前嵌入余弦差值未提供'
  const denseTop = value.dense_top_item_id || value.ranking_separation?.dense_top_item_id
  const finalTop = value.final_top_item_id || value.ranking_separation?.final_top_item_id
  const winnerText = denseTop && finalTop && denseTop !== finalTop ? `；嵌入首名 ${denseTop} 与最终首名 ${finalTop} 不同，以上差值不描述最终排序差距` : ''
  const signal = value.signal_agreement
  const status = typeof signal === 'object' ? signal?.status : signal
  const agreementLabels: Record<string, string> = {agree: '已比较信号一致', disagree: '已比较信号不一致', mixed: '已比较信号部分一致', not_measured: '信号一致性未测量', available: '信号可用性记录', partial: '信号部分可用'}
  const agreement = agreementLabels[status || ''] || '信号一致性未提供'
  const count = typeof signal === 'object' && signal.compared ? `（${signal.agreeing ?? 0}/${signal.compared}）` : ''
  const probe = value.signal_availability?.probes_applied === false ? '；属性探针未参与排序' : ''
  return `${coverageText}；${marginText}${winnerText}；${agreement}${count}${probe}。这些诊断不表示结果正确率。`
}

export default function App() {
  const [query, setQuery] = useState('')
  const [variant, setVariant] = useState<Variant>('image_only')
  const [searchMode, setSearchMode] = useState<SearchMode>('text')
  const [gallery, setGallery] = useState<'frozen' | 'expanded'>('frozen')
  const [visualMethod, setVisualMethod] = useState<VisualMethod>('siglip')
  const [aggregation, setAggregation] = useState<'mean' | 'max' | 'centroid'>('mean')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState('')
  const [compare, setCompare] = useState(false)
  const [topK, setTopK] = useState(12)
  const [filters, setFilters] = useState<Filters>({ object_type: [], material: [], era: [-5000, 2026], source: [] })
  const [facets, setFacets] = useState<Facets>({})
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [baselineResponse, setBaselineResponse] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [compareLoading, setCompareLoading] = useState(false)
  const [error, setError] = useState('')
  const [detail, setDetail] = useState<Result | null>(null)
  const [detailData, setDetailData] = useState<Record<string, unknown> | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [showMethod, setShowMethod] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [facetError, setFacetError] = useState('')
  const [searchSeq] = useState(() => ({ current: 0 }))
  const imageRoute = searchMode !== 'text'
  const effectiveCompare = compare && !imageRoute
  const expandedCount = typeof health?.expanded_catalogue_count === 'number' ? health.expanded_catalogue_count : 0
  const textSearchEndpoint = gallery === 'expanded' && expandedCount > 4692 ? '/api/search/expanded' : '/api/search'

  useEffect(() => {
    if (!imageFile) { setImagePreview(''); return }
    const url = URL.createObjectURL(imageFile)
    setImagePreview(url)
    return () => URL.revokeObjectURL(url)
  }, [imageFile])

  const changeMode = (mode: SearchMode) => {
    ++searchSeq.current; setSearchMode(mode); setResponse(null); setBaselineResponse(null)
    setError(''); setLoading(false); setCompareLoading(false)
  }

  useEffect(() => { fetch('/api/facets').then(r => { if (!r.ok) throw new Error(`facets ${r.status}`); return r.json() }).then((data: Facets) => { setFacets(data) }).catch(() => setFacetError('筛选统计暂时不可用，仍可直接搜索。')); fetch('/api/health').then(r => r.ok ? r.json() : Promise.reject()).then(setHealth).catch(() => setHealth({status:'offline'})) }, [])

  const doSearch = async (forcedVariant = variant, suppliedQuery?: string) => {
    const text = (suppliedQuery ?? query).trim()
    if (searchMode !== 'image' && !text) { setError('请输入中文或英文查询，例如“明代青花瓷瓶，上面有龙纹”。'); return }
    if (searchMode === 'image' && !imageFile) { setError('请选择一张参考图片。'); return }
    if (searchMode === 'image' && imageFile && imageFile.size > 12 * 1024 * 1024) { setError('参考图片最大为 12 MB。'); return }
    const seq = ++searchSeq.current; setError(''); setLoading(true); setResponse(null); setBaselineResponse(null)
    const payload = { query: text, top_k: Math.min(topK, 50), variant: forcedVariant, filters: cleanFilters(filters), stream: false }
    try {
      const primary = searchMode === 'image'
        ? fetch(`/api/search/image?${new URLSearchParams({method: visualMethod, top_k: String(Math.min(topK, 50))})}`, {method: 'POST', headers: {'Content-Type': imageFile?.type || 'application/octet-stream'}, body: imageFile})
        : searchMode === 'reference'
          ? fetch('/api/search/reference', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: text, top_k: Math.min(topK, 50), method: visualMethod, aggregation, limit: 4})})
          : fetch(textSearchEndpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      const requests = [primary]
      if (effectiveCompare) requests.push(fetch(textSearchEndpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...payload, variant: 'baseline' }) }))
      const replies = await Promise.all(requests)
      if (seq !== searchSeq.current) return
      if (!replies[0].ok) { const failure = await replies[0].json().catch(() => null); throw new Error(typeof failure?.detail === 'string' ? failure.detail : `搜索服务返回 ${replies[0].status}`) }
      const data = await replies[0].json() as SearchResponse
      if (seq !== searchSeq.current) return
      setResponse({...data, query: searchMode === 'image' ? imageFile?.name || '上传参考图' : data.query})
      if (effectiveCompare) { setCompareLoading(true); if (replies[1]?.ok) setBaselineResponse(await replies[1].json() as SearchResponse); else setError('基线检索失败，请稍后重试。') }
    } catch (e) { if (seq === searchSeq.current) setError(e instanceof Error ? `${e.message}。请检查方法可用性或服务状态。` : '暂时无法连接搜索服务。') }
    finally { if (seq === searchSeq.current) { setLoading(false); setCompareLoading(false) } }
  }
  const cleanFilters = (f: Filters) => ({ object_type: f.object_type.length ? f.object_type : undefined, material: f.material.length ? f.material : undefined, era: f.era[0] > -5000 || f.era[1] < 2026 ? f.era : undefined, source: f.source.length ? f.source : undefined })

  const openDetail = async (item: Result) => {
    setDetail(item); setDetailData(null); setDetailLoading(true)
    try { const r = await fetch(`/api/items/${item.item_id}`); if (r.ok) setDetailData(await r.json()) } catch { /* detail remains with search metadata */ } finally { setDetailLoading(false) }
  }
  const parsedEntries = useMemo(() => Object.entries(response?.parsed || {}).filter(([, v]) => v !== null && v !== undefined && v !== '' && (!Array.isArray(v) || v.length)), [response])
  const filterOptions = facets.object_type?.length ? Object.fromEntries(facets.object_type.map(x => [x.value, x.count])) : { vase: 0, bowl: 0, jar: 0, plate: 0, vessel: 0 }
  const materialOptions = facets.material?.length ? Object.fromEntries(facets.material.map(x => [x.value, x.count])) : { porcelain: 0, bronze: 0, jade: 0, ceramic: 0, textile: 0 }
  const hasResults = !!response && Array.isArray(response.results)

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><span>拾</span></div><div><div className="brand-title">拾遗 <i>Curio</i></div><div className="brand-sub">跨语言文物图像检索</div></div></div>
      <nav className="topnav"><a href="#search">检索工作台</a><a href="#about">项目方法</a><span className={`status-dot ${health?.status === 'ok' ? 'online' : ''}`}><span /> {health?.status === 'offline' ? 'API 未连接' : health ? `API ${health.status === 'ok' ? '在线' : '降级'} · ${health.items ?? '—'} 件 · 索引 ${health.indexed_images ?? '—'} 张 · ${health.index_version || '未知版本'}` : '正在检查 API'}</span></nav>
      <button className="icon-button mobile-menu" aria-label="菜单"><Menu size={21}/></button>
    </header>

    <main>
      <section className="hero" id="search">
        <div className="hero-copy"><p className="eyebrow"><Sparkles size={14}/> MUSEUM OPEN ACCESS · 公开馆藏</p><h1>用一句话，寻找<br/><em>中国古代文物。</em></h1><p className="hero-lead">输入中文或英文的自然语言，检索博物馆开放藏品。结果同时显示图文相似度与馆藏字段依据；当前结果尚未进行视觉核验。</p></div>
        <div className="search-card">
          <div className="search-mode-tabs" role="group" aria-label="检索输入方式">{([{id: 'text', label: '文本检索'}, {id: 'reference', label: '文字寻找参考图'}, {id: 'image', label: '上传图片检索'}] as {id: SearchMode; label: string}[]).map(mode => <button key={mode.id} className={searchMode === mode.id ? 'active' : ''} aria-pressed={searchMode === mode.id} onClick={() => changeMode(mode.id)}>{mode.label}</button>)}</div>
          <div className="gallery-picker"><label htmlFor="search-gallery">检索图库</label><select id="search-gallery" value={imageRoute ? 'frozen' : gallery} disabled={imageRoute} onChange={e => {++searchSeq.current; setGallery(e.target.value as typeof gallery); setResponse(null); setBaselineResponse(null); setError(''); setLoading(false); setCompareLoading(false)}}><option value="frozen">冻结图库 · 4,692 件</option><option value="expanded" disabled={expandedCount <= 4692}>{expandedCount > 4692 ? `扩展图库 · ${expandedCount.toLocaleString()} 件（研究）` : '扩展图库暂不可用'}</option></select><p>{imageRoute ? '参考图与上传图片检索使用冻结图库。' : gallery === 'expanded' && expandedCount > 4692 ? `新增 ${expandedCount - 4692} 件研究样本，未纳入原有评测；基线对比使用同一扩展图库。筛选统计仍来自冻结图库。` : '使用原有冻结图库与评测范围。'}</p></div>
          {searchMode === 'image' ? <div className="image-upload-panel"><label htmlFor="query-image" className="upload-label">选择参考图片 <span>最大 12 MB · 24 百万像素</span></label><input id="query-image" type="file" accept="image/*" onChange={e => {++searchSeq.current; setImageFile(e.target.files?.[0] || null); setError(''); setResponse(null); setLoading(false)}}/>{imageFile && <div className="uploaded-preview">{imagePreview && <img src={imagePreview} alt="上传的检索参考图"/>}<span>{imageFile.name}<small>{(imageFile.size / 1024 / 1024).toFixed(2)} MB</small></span></div>}<button className="search-submit upload-submit" onClick={() => doSearch()} disabled={loading || !imageFile}>{loading ? '检索中…' : '用此图片检索'}<ArrowUpRight size={17}/></button></div> : <>
          <div className="search-input-wrap"><Search size={21}/><input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && doSearch()} placeholder="例如：明代青花瓷瓶，上面有龙纹" aria-label="搜索馆藏"/><button className="clear-search" onClick={() => setQuery('')} aria-label="清空">{query && <X size={16}/>}</button><button className="search-submit" onClick={() => doSearch()} disabled={loading}><span>{loading ? '检索中…' : '开始检索'}</span><ArrowUpRight size={17}/></button></div>
          <div className="example-row"><span>试试：</span>{(searchMode === 'reference' ? referenceExamples : examples).map(x => <button key={x} onClick={() => { setQuery(x); }}>{x}</button>)}</div>
          </>}
          {imageRoute && <div className="visual-controls"><label>图像比较方法<select value={visualMethod} onChange={e => setVisualMethod(e.target.value as VisualMethod)}>{visualMethods.map(method => <option key={method.id} value={method.id}>{method.label}</option>)}</select></label>{searchMode === 'reference' && <label>多参考图汇总<select value={aggregation} onChange={e => setAggregation(e.target.value as typeof aggregation)}><option value="mean">逐图相似度均值</option><option value="max">逐图相似度最大值</option><option value="centroid">平均向量</option></select></label>}<p>{searchMode === 'reference' ? '从馆外参考图库选取最多 4 张图片，再按图像特征检索本地图库。下方显示实际参考图与选择计划。' : '使用上传图片与本地图像比较。'} 本地候选排序不使用器型、材质、年代或来源筛选。形状、颜色、纹理可能因前景不可靠而弃权；融合权重属于研究配置。</p></div>}
        </div>
      </section>

      <section className="workspace">
        <aside className={`filters-panel ${showFilters ? 'open' : ''}`}>
          <div className="panel-title"><span><SlidersHorizontal size={17}/> 筛选馆藏</span><button className="mobile-close" onClick={() => setShowFilters(false)}><X size={18}/></button></div>
          {imageRoute && <p className="filters-unused"><Info size={15}/> 当前图像比较路径不使用元数据筛选；已选条件将保留供文本检索使用。</p>}
          <fieldset className="filter-fields" disabled={imageRoute} aria-label="馆藏元数据筛选">
          <FilterGroup title="器型" options={filterOptions} selected={filters.object_type} onToggle={v => toggleFilter('object_type', v, filters, setFilters)} />
          <FilterGroup title="材质" options={materialOptions} selected={filters.material} onToggle={v => toggleFilter('material', v, filters, setFilters)} />
          <div className="filter-group"><div className="filter-heading">年代 <span>{filters.era[0] <= -5000 ? '最早' : filters.era[0]} — {filters.era[1] === 2026 ? '至今' : filters.era[1]}</span></div><div className="range-row"><input type="number" value={filters.era[0]} placeholder="起始" onChange={e => setFilters({...filters, era: [e.target.value === '' ? -5000 : Number(e.target.value), filters.era[1]]})}/><span>至</span><input type="number" value={filters.era[1] === 2026 ? '' : filters.era[1]} placeholder="结束" onChange={e => setFilters({...filters, era: [filters.era[0], Number(e.target.value) || 2026]})}/></div></div>
          <FilterGroup title="数据来源" options={facets.source?.length ? Object.fromEntries(facets.source.map(x => [x.value, x.count])) : { 'Met Open Access': 0 }} selected={filters.source} onToggle={v => toggleFilter('source', v, filters, setFilters)} />
          <button className="reset-button" onClick={() => setFilters({ object_type: [], material: [], era: [-5000, 2026], source: [] })}>重置筛选</button>
          </fieldset>
          <div className="license-note"><Database size={15}/><span>图片与馆藏信息来自开放数据。详情中的字段均标注为“馆藏元数据”。</span></div>
        </aside>

        <section className="results-area">
          <div className="results-toolbar"><button className="filter-mobile-button" onClick={() => setShowFilters(true)}><Filter size={16}/> 筛选</button><div className="toolbar-left"><div className="result-heading">{hasResults ? <><strong>{response?.results?.length || 0}</strong> 条结果</> : '准备开始检索'}{response?.query && <span className="query-chip">“{response.query}”</span>}</div>{response?.degraded && <span className="degraded-tag"><Info size={13}/> 已降级</span>}</div><div className="toolbar-right"><label className="topk-label">显示 <select value={topK} onChange={e => setTopK(Number(e.target.value))}><option value={8}>8</option><option value={12}>12</option><option value={24}>24</option><option value={48}>48</option></select> 条</label><button className={`compare-toggle ${effectiveCompare ? 'active' : ''}`} disabled={imageRoute} title={imageRoute ? '图像比较路径不提供文本基线对比' : undefined} onClick={() => setCompare(!compare)}><Layers3 size={16}/> {effectiveCompare ? '关闭对比' : '对比基线'}</button></div></div>
          {facetError && <div className="alert warning"><Info size={17}/><span>{facetError}</span></div>}
          <div className="method-strip">{imageRoute ? <span className="active-image-method"><Zap size={15}/> 当前图像方法：<b>{visualMethods.find(method => method.id === visualMethod)?.label}</b></span> : <><button onClick={() => setShowMethod(!showMethod)}><Zap size={15}/> 当前方法：<b>{variants.find(v => v.id === variant)?.label}</b><ChevronDown size={14} className={showMethod ? 'rotate' : ''}/></button>{showMethod && <div className="method-menu">{variants.map(v => <button key={v.id} className={variant === v.id ? 'selected' : ''} onClick={() => {setVariant(v.id); setShowMethod(false)}}><span><b>{v.label}</b><small>{v.note}</small></span>{variant === v.id && <span className="check">✓</span>}</button>)}</div>}</>}{response?.effective_variant && response.effective_variant !== (imageRoute ? visualMethod : variant) && <span className="effective-method">实际路径：{response.effective_variant}</span>}<span className="method-help">方法选择只影响下一次检索</span></div>
          {error && <div className="alert error"><Info size={17}/><span>{error}</span><button onClick={() => setError('')}><X size={15}/></button></div>}
          {response?.warnings?.map((w, i) => <div className="alert warning" key={i}><Info size={17}/><span>{w}</span></div>)}
          {response && <ReferencePanel references={response.references} plan={response.plan}/>}
          {response?.patch_diagnostics && <PatchDiagnostics value={response.patch_diagnostics}/>}
          {response && <ConfidenceStrip value={response.confidence} />}
          {!loading && !error && !hasResults && searchMode === 'image' && <div className="empty-state"><div className="empty-icon"><Search size={25}/></div><h2>用图片寻找相似藏品</h2><p>选择参考图片与图像比较方法，然后开始检索。</p></div>}
          {!loading && !error && !hasResults && searchMode !== 'image' && <EmptyState reference={searchMode === 'reference'} onExample={x => {setQuery(x); doSearch(variant, x)}} />}
          {loading && <LoadingState />}
          {!loading && hasResults && (effectiveCompare ? <div className="comparison-grid"><ResultColumn title={variants.find(v => v.id === variant)?.label || '当前方法'} subtitle={variants.find(v => v.id === variant)?.note || ''} data={response} onOpen={openDetail}/><ResultColumn title="基线" subtitle="单路图文相似度" data={baselineResponse} onOpen={openDetail} loading={compareLoading}/></div> : <ResultGrid data={response} onOpen={openDetail}/>) }
          {response && hasResults && <div className="timing-footer"><span><Clock3 size={14}/> 总耗时 {formatTime(response.timings?.total_ms)}</span><span>解析 {formatTime(response.timings?.parse_ms)}</span><span>编码 {formatTime(response.timings?.encode_ms)}</span><span>召回 {formatTime(response.timings?.retrieve_ms)}</span><span>融合 {formatTime(response.timings?.fusion_ms)}</span><span>重排 {formatTime(response.timings?.rerank_ms)}</span><span>暖机总耗时 {formatTime(response.timings?.warm_total_ms)}</span><span>冷启动加载 {formatTime(response.timings?.model_load_ms || response.timings?.cold_start_ms)}</span><span className="version">{response.model_version || '模型版本未提供'} · {response.index_version || '索引版本未提供'}</span></div>}
        </section>
      </section>
    </main>

    <footer id="about"><div><span className="footer-name">拾遗 Curio</span><span>课程项目 · 文本到图像检索</span></div><div className="footer-links"><span><BookOpen size={14}/> 以接口返回依据为准</span><span><Globe2 size={14}/> 中文查询 · English queries</span></div></footer>
    {detail && <DetailDrawer item={detail} data={detailData} loading={detailLoading} onClose={() => setDetail(null)} />}
  </div>
}

function toggleFilter(key: keyof Filters, value: string, filters: Filters, setFilters: React.Dispatch<React.SetStateAction<Filters>>) { const current = filters[key] as string[]; setFilters({...filters, [key]: current.includes(value) ? current.filter(v => v !== value) : [...current, value]}) }
function FilterGroup({title, options, selected, onToggle}: {title: string; options: Record<string, number>; selected: string[]; onToggle: (v: string) => void}) { return <div className="filter-group"><div className="filter-heading">{title}</div><div className="check-list">{Object.entries(options).map(([key, count]) => <label className={`check-row ${selected.includes(key) ? 'checked' : ''}`} key={key}><input type="checkbox" checked={selected.includes(key)} onChange={() => onToggle(key)}/><span className="custom-check">✓</span><span>{key}</span><small>{count || '—'}</small></label>)}</div></div> }
function EmptyState({onExample, reference = false}: {onExample: (q: string) => void; reference?: boolean}) { return <div className="empty-state"><div className="empty-icon"><Search size={25}/></div><h2>{reference ? '先寻找馆外参考图' : '从一个问题开始'}</h2><p>{reference ? '用朝代、年号与器型选择参考图，再比较图片。参考图来源记录不能验证本地结果年代。' : '描述器型、材质、纹饰或年代。系统会把查询拆成槽位，并逐条解释结果。'}</p><div className="empty-examples">{(reference ? referenceExamples : examples).map(x => <button key={x} onClick={() => onExample(x)}>{x}<ArrowUpRight size={14}/></button>)}</div></div> }
function LoadingState() { return <div className="loading-state"><div className="loading-orbit"><span/><span/><span/></div><h2>正在检索开放馆藏</h2><p>解析查询 · 召回候选 · 尚未进行视觉核验</p><div className="skeleton-grid">{[1,2,3,4].map(x => <div className="skeleton-card" key={x}><div/><span/><i/></div>)}</div></div> }
function ResultGrid({data, onOpen}: {data: SearchResponse; onOpen: (r: Result) => void}) { return <div className="result-grid">{(data.results || []).length ? (data.results || []).map(r => <ResultCard key={String(r.item_id)} result={r} onOpen={onOpen}/>) : <div className="no-results">没有找到结果。试试更宽泛的查询或清除筛选条件。</div>}</div> }
function ResultColumn({title, subtitle, data, onOpen, loading}: {title: string; subtitle: string; data: SearchResponse | null; onOpen: (r: Result) => void; loading?: boolean}) { return <div className="result-column"><div className="column-head"><div><h3>{title}</h3><span>{subtitle}</span></div>{data && <strong>{data.results?.length || 0} 条</strong>}</div>{loading ? <LoadingState/> : data ? <div className="column-results">{(data.results || []).map(r => <ResultCard key={String(r.item_id)} result={r} onOpen={onOpen} compact/>)}</div> : <div className="column-pending"><Clock3 size={18}/> 等待基线返回…</div>}</div> }
function ReferencePanel({references, plan}: {references?: Reference[]; plan?: Record<string, unknown>}) {
  if (!plan && !references?.length) return null
  const terms = Array.isArray(plan?.query_terms) ? plan.query_terms.map(String) : []
  const eras = Array.isArray(plan?.required_eras) ? plan.required_eras.map(String) : []
  const searches = Array.isArray(plan?.museum_queries) ? plan.museum_queries as {source?: string; query?: string}[] : []
  return <section className="reference-panel" aria-label="参考图片与查询计划">
    {plan && <div className="reference-plan"><div className="reference-panel-heading"><h2>参考图选择计划</h2>{plan.is_llm === false && <span>双语词汇规则</span>}</div><p>先按查询选择馆外参考图，再用这些图片检索本地图像。</p>{terms.length > 0 && <div className="plan-terms"><b>检索词</b>{terms.map(term => <span key={term}>{term}</span>)}</div>}{eras.length > 0 && <p className="plan-eras">参考图年代条件：{eras.join('、')}（来自来源记录；不验证本地结果年代）</p>}{Boolean(plan.unmapped) && <p className="plan-empty">当前规则未覆盖此查询，没有可用参考检索词。</p>}{searches.length > 0 && <details><summary>查看馆藏来源检索计划（{searches.length} 项）</summary><ul>{searches.map((entry, index) => <li key={index}><b>{entry.source || '来源未提供'}</b><span>{entry.query}</span></li>)}</ul></details>}</div>}
    <div className="reference-panel-heading"><h2>本次选用参考图</h2><span>{references?.length || 0} 张</span></div>
    {references?.length ? <div className="reference-grid">{references.map(reference => <article className="reference-card" key={reference.reference_id}><img src={`/api/references/${encodeURIComponent(reference.reference_id)}/image`} alt={reference.title || '馆外参考图片'} loading="lazy"/><div><h3>{reference.title || reference.reference_id}</h3><p>{reference.source || '来源未提供'}</p>{[reference.date, reference.period, reference.reign].filter(Boolean).length > 0 && <p className="reference-dates">{Array.from(new Set([reference.date, reference.period, reference.reign].filter(Boolean))).join(' · ')}</p>}{reference.license && <small>{reference.license}</small>}{reference.source_url && <a href={reference.source_url} target="_blank" rel="noreferrer">来源记录 <ExternalLink size={12}/></a>}</div></article>)}</div> : <p className="plan-empty">未选到参考图。可改写查询或切换到上传图片检索。</p>}
  </section>
}
function PatchDiagnostics({value}: {value: NonNullable<SearchResponse['patch_diagnostics']>}) {
  const reranked = value.status === 'reranked'
  return <div className="patch-diagnostics" role="status"><strong>DINO 局部特征研究诊断 · {reranked ? '已进行局部特征重排' : value.status === 'fallback_cls' ? '前景证据不足，使用 CLS 排序' : '状态未提供'}</strong><p>CLS 召回 {value.recall_count ?? '—'} 件；前景可用候选 {value.accepted_candidates ?? '—'} 件；前景可用参考图 {value.accepted_references ?? '—'}/{value.total_references ?? '—'} 张。</p><p>先用 DINO CLS 召回，再比较局部特征与软轮廓。前景不可靠的候选保留原有位置；这些统计不表示语义纹饰（如龙纹）识别或结果正确率。</p></div>
}
function ConfidenceStrip({value}: {value?: SearchResponse['confidence']}) {
  const level = value?.level || 'unknown'
  return <div className={`confidence-strip confidence-${level}`}><div><strong>检索诊断：{value?.label || '无法判断'}</strong><span>{confidenceText(value)}</span></div><small>{value?.basis || '这些诊断未经正确率校准，不代表人工确认。'}</small></div>
}
function ResultCard({result, onOpen, compact}: {result: Result; onOpen: (r: Result) => void; compact?: boolean}) { const slots = (result.matched_slots || []).map(x => [x.slot, true, x.value] as [string, boolean | null, string | undefined]); return <article className={`result-card ${compact ? 'compact' : ''}`} onClick={() => onOpen(result)}><div className="image-wrap"><img src={result.thumb_url || `/api/images/${result.item_id}?thumb=true`} alt={result.title || '馆藏图像'} loading="lazy" onError={e => { e.currentTarget.style.display = 'none' }}/><span className="rank">#{result.rank}</span><span className="score">{formatScore(result.score)}</span></div><div className="card-body"><div className="card-meta"><span>{result.source || '馆藏来源未提供'}</span><span>馆藏元数据</span></div><h3>{result.title || `馆藏对象 ${result.item_id}`}</h3>{result.evidence && <p className="evidence">{Array.isArray(result.evidence) ? result.evidence.map(evidenceText).join('；') : String(result.evidence)}</p>}<div className="slot-row">{slots.slice(0, compact ? 3 : 5).map(([k, v, value]) => <span className={`slot ${v === true ? 'yes' : v === false ? 'no' : 'unknown'}`} key={`${k}:${value || ''}`}><i>{v === true ? '✓' : v === false ? '×' : '?'}</i>{readableSlot(k)}</span>)}</div><div className="card-foot"><span>排序分数 {formatScore(result.score)}</span><button onClick={e => {e.stopPropagation(); onOpen(result)}}>查看解释 <ChevronRight size={15}/></button></div></div></article> }
function cleanDescription(value: unknown) {
  return String(value ?? '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
}
function detailValue(key: string, value: unknown) {
  if (key === 'description') return cleanDescription(value)
  if (typeof value === 'number') return Number.isFinite(value) ? value.toString() : '—'
  if (Array.isArray(value)) return value.map(item => typeof item === 'object' ? JSON.stringify(item) : String(item)).join(', ')
  if (value && typeof value === 'object') return JSON.stringify(value)
  return String(value ?? '')
}
function DetailDrawer({item, data, loading, onClose}: {item: Result; data: Record<string, unknown> | null; loading: boolean; onClose: () => void}) {
  const merged = {...item, ...(data || {})} as Record<string, unknown>
  const primaryKeys = ['title', 'source', 'medium', 'culture', 'period', 'year_from', 'year_to', 'tags', 'description', 'license', 'credit_line', 'object_type', 'material', 'motifs', 'colors']
  const technicalKeys = ['item_id', 'source_id', 'split', 'sha256', 'phash', 'phash_group', 'duplicate_group', 'mapping_basis', 'origin_basis', 'image_path', 'thumb_path']
  const entries = primaryKeys.map(key => [key, merged[key]] as [string, unknown]).filter(([, value]) => value !== null && value !== undefined && value !== '')
  const technical = technicalKeys.map(key => [key, merged[key]] as [string, unknown]).filter(([, value]) => value !== null && value !== undefined && value !== '')
  const scoreBreakdown = Object.entries((merged.score_breakdown as Record<string, unknown>) || {})
  return <div className="drawer-backdrop" onClick={onClose}><aside className="detail-drawer" onClick={e => e.stopPropagation()}>
    <div className="drawer-head"><div><span className="eyebrow">馆藏对象 · {String(item.item_id)}</span><h2>{String(merged.title || `馆藏对象 ${item.item_id}`)}</h2></div><button className="icon-button" onClick={onClose}><X size={19}/></button></div>
    <div className="drawer-image"><img src={String(merged.image_url || merged.thumb_url || `/api/images/${item.item_id}`)} alt={String(merged.title || '')} onError={e => { e.currentTarget.style.display = 'none' }}/></div>
    <div className="drawer-section"><h3>为什么匹配</h3><p className="drawer-evidence">{Array.isArray(merged.evidence) ? merged.evidence.map((entry) => evidenceText(entry as {kind?: string; text?: string; value?: string; probability?: number})).join('；') || '后端未提供逐项证据。' : String(merged.evidence || '后端未提供逐项证据。')}</p><div className="slot-explain">{(Array.isArray(merged.matched_slots) ? merged.matched_slots : []).map((entry) => { const k = String((entry as {slot?: string}).slot || ''); const value = (entry as {value?: string}).value || ''; return <div key={`${k}:${value}`}><span className="slot-dot yes">✓</span><span>{readableSlot(k)}{value ? ` · ${value}` : ''}</span><b>命中</b></div> })}</div></div>
    {Boolean(merged.confidence) && <ConfidenceStrip value={merged.confidence as SearchResponse['confidence']} />}
    {loading && <div className="drawer-loading">正在读取对象详情…</div>}
    <div className="drawer-section"><h3>馆藏元数据</h3><div className="metadata-list">{entries.map(([key, value]) => <div key={key}><span>{readableSlot(key)}</span><b>{detailValue(key, value)}</b></div>)}</div></div>
    {technical.length > 0 && <details className="drawer-section technical-details"><summary>技术溯源字段</summary><div className="metadata-list">{technical.map(([key, value]) => <div key={key}><span>{readableSlot(key)}</span><b>{detailValue(key, value)}</b></div>)}</div></details>}
    {scoreBreakdown.length > 0 && <div className="drawer-section"><h3>排序分解</h3><div className="metadata-list">{scoreBreakdown.map(([key, value]) => <div key={key}><span>{readableSlot(key)}</span><b>{detailValue(key, value)}</b></div>)}</div></div>}
    {Boolean(merged.source_url) && <a className="source-link" href={String(merged.source_url)} target="_blank" rel="noreferrer">打开来源页面 <ExternalLink size={15}/></a>}
    <div className="drawer-note"><Info size={15}/> 证据来自检索模型；标题、年代、材质等字段来自“馆藏元数据”，不作为图像可见性结论。</div>
  </aside></div>
}
