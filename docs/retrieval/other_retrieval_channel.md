# 其他召回通道

  - [ ] 其他召回通道
    - [ ] GeoHahs
      - GeoHash：对经纬度的编码，地图上⼀个长⽅形区域
      - 索引：GeoHash->优质笔记列表（按时间倒排）
    - [ ] 同城召回
      - 索引：城市->优质笔记列表（按时间倒排）
    - [ ] 关注作者召回
      - ⽤户->关注的作者->最新的笔记
    - [ ] 有交互作者召回
      - ⽤户->有交互的作者->最新的笔记
    - [ ] 相似作者召回
      - ⽤户->感兴趣的作者->相似作者->最新的笔记
    - [ ] 缓存召回
      - 精排前50，但是没有曝光的，缓存起来，作为⼀条召回通道。
      - 缓存⼤⼩固定，需要退场机制。
        - 召回次数、保存天数、FIFO、曝光退场