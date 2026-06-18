import re

with open('c:/Users/ADMIN/Downloads/DesNet121/mdep_densenet_notebook.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. ISICDataset __init__ and __getitem__
code = re.sub(
    r'def __init__\(self, dataframe, image_dir, transform=None, hdf5_path=None\):(.*?)self\._error_printed = False',
    r'def __init__(self, dataframe, image_dir, tabular_cols=None, transform=None, hdf5_path=None):\1self.tabular_cols = tabular_cols or []\n        self._error_printed = False',
    code, flags=re.DOTALL
)

getitem_replacement = r'''    def __getitem__(self, idx):
        isic_id = self.data_frame.iloc[idx]['isic_id']
        image = None

        if self.hdf5_path and getattr(self, 'HAS_H5PY', globals().get('HAS_H5PY', False)):
            try:
                hf = self._get_hdf5()
                if hf and isic_id in hf:
                    img_bytes = hf[isic_id][()]
                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception:
                pass

        if image is None and self.image_dir:
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (384, 384), color='black')

        target = self.data_frame.iloc[idx]['target']
        
        # EXTRACT TABULAR METADATA
        if self.tabular_cols:
            tabular = self.data_frame.iloc[idx][self.tabular_cols].values.astype(np.float32)
            tabular_tensor = torch.tensor(tabular)
        else:
            tabular_tensor = torch.zeros(1, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, tabular_tensor, torch.tensor(target, dtype=torch.long)'''

code = re.sub(
    r'    def __getitem__\(self, idx\):.*?return image, torch\.tensor\(target, dtype=torch\.long\)',
    getitem_replacement,
    code, flags=re.DOTALL
)

# 2. get_isic_dataloaders signature and return
code = re.sub(
    r'def get_isic_dataloaders\(batch_size=32, test_ratio=0\.2\):',
    r'def get_isic_dataloaders(batch_size=32, test_ratio=0.2):\n    tabular_cols = []',
    code
)

dummy_replacement = r'''        X = torch.randn(200, 3, 384, 384)
        T = torch.randn(200, 5)
        Y = torch.randint(0, 2, (200,))
        full = TensorDataset(X, T, Y)
        tr = Subset(full, range(160))
        te = Subset(full, range(160, 200))
        return (DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(te, batch_size=batch_size),
                num_classes,
                torch.ones(num_classes),
                5)'''
code = re.sub(
    r'        X = torch\.randn\(200, 3, 224, 224\).*?torch\.ones\(num_classes\)\)  # uniform weights for dummy data',
    dummy_replacement,
    code, flags=re.DOTALL
)

code = code.replace('transforms.Resize((224, 224))', 'transforms.Resize((384, 384))')

split_replacement = r'''    # --- Multi-Modal Tabular Preprocessing ---
    df_tab = df[['age_approx', 'sex', 'anatom_site_general_challenge', 'clin_size_long_diam_mm']].copy() if 'age_approx' in df.columns else pd.DataFrame()
    if not df_tab.empty:
        df_tab['age_approx'] = df_tab['age_approx'].fillna(df_tab['age_approx'].median())
        if 'clin_size_long_diam_mm' in df_tab.columns:
            df_tab['clin_size_long_diam_mm'] = df_tab['clin_size_long_diam_mm'].fillna(df_tab['clin_size_long_diam_mm'].median())
        for col in ['sex', 'anatom_site_general_challenge']:
            if col in df_tab.columns:
                df_tab[col] = df_tab[col].fillna(df_tab[col].mode()[0] if not df_tab[col].mode().empty else 'unknown')
        
        cat_cols = [c for c in ['sex', 'anatom_site_general_challenge'] if c in df_tab.columns]
        df_tab = pd.get_dummies(df_tab, columns=cat_cols, drop_first=True)
        
        num_cols = [c for c in ['age_approx', 'clin_size_long_diam_mm'] if c in df_tab.columns]
        from sklearn.preprocessing import StandardScaler
        if num_cols:
            scaler = StandardScaler()
            df_tab[num_cols] = scaler.fit_transform(df_tab[num_cols])
            
        tabular_cols = df_tab.columns.tolist()
        for col in tabular_cols:
            df[col] = df_tab[col]

    # --- Eradicate Data Leakage via Strict Patient-Level Cross Validation ---
    from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
    
    if 'patient_id' in df.columns:
        print('🧬 Found patient_id. Using GroupShuffleSplit to prevent data leakage.')
        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        train_idx, test_idx = next(gss.split(df, df['target'], groups=df['patient_id']))
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
    else:
        print('⚠ patient_id not found. Falling back to StratifiedKFold.')
        skf = StratifiedKFold(n_splits=int(1/test_ratio), shuffle=True, random_state=42)
        train_idx, test_idx = next(skf.split(df, df['target']))
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

    print(f'📊 Train: {len(train_df)} samples  |  Test: {len(test_df)} samples')
    train_ds = ISICDataset(train_df, image_dir, tabular_cols=tabular_cols, transform=train_tf, hdf5_path=hdf5_path)
    test_ds  = ISICDataset(test_df,  image_dir, tabular_cols=tabular_cols, transform=test_tf,  hdf5_path=hdf5_path)'''

code = re.sub(
    r'    train_df, test_df = train_test_split\([\s\S]*?test_ds  = ISICDataset\(test_df,  image_dir, transform=test_tf,  hdf5_path=hdf5_path\)',
    split_replacement,
    code
)

code = re.sub(
    r'return train_loader, test_loader, num_classes, cw$',
    r'return train_loader, test_loader, num_classes, cw, len(tabular_cols)',
    code, flags=re.MULTILINE
)

multimodal_model = r'''class MultimodalDenseNet(nn.Module):
    """Multi-Modal Architecture: Fuses DenseNet-121 image features with tabular metadata."""
    def __init__(self, backbone, tab_features, num_classes):
        super().__init__()
        self.features = backbone.features
        self.use_tabular = tab_features > 0
        in_features = backbone.classifier.in_features
        
        if self.use_tabular:
            self.tab_mlp = nn.Sequential(
                nn.Linear(tab_features, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_features + 64, num_classes),
                EvidenceLayer(activation='softplus')
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes),
                EvidenceLayer(activation='softplus')
            )

    def forward(self, img_x, tab_x=None):
        features = self.features(img_x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        img_feats = torch.flatten(out, 1)
        
        if self.use_tabular and tab_x is not None:
            tab_out = self.tab_mlp(tab_x)
            fuse = torch.cat([img_feats, tab_out], dim=1)
            return self.classifier(fuse)
        return self.classifier(img_feats)'''

code = code.replace('def compute_ece(', multimodal_model + '\n\ndef compute_ece(')

main_model_init = r'''    train_loader, test_loader, num_classes, class_weights, tabular_dim = get_isic_dataloaders(batch_size=32)
    
    # HÀNH ĐỘNG 1: Can thiệp Data Loader (WeightedRandomSampler)
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'data_frame'):
        train_targets = train_dataset.data_frame['target'].values
    else:
        train_targets = np.array([t.item() for _, _, t in train_dataset]) # tabular tuple
        
    class_sample_counts = np.array([np.sum(train_targets == 0), np.sum(train_targets == 1)])
    weight_per_class = 1.0 / class_sample_counts
    samples_weights = np.array([weight_per_class[t] for t in train_targets])
    samples_weights = torch.from_numpy(samples_weights).double()
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=sampler, 
        num_workers=4, pin_memory=True, drop_last=True
    )

    print(f"📊 Classes: {num_classes}")

    # ── Model: DenseNet-121 with Multi-Modal EDL head ──────────────────────────
    base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model = MultimodalDenseNet(base_model, tabular_dim, num_classes)
    
    nn.init.normal_(model.classifier[0].weight, mean=0, std=0.001)
    
    with torch.no_grad():
        model.classifier[0].bias[0] = 0.0
        model.classifier[0].bias[1] = 0.0'''

code = re.sub(
    r'    train_loader, test_loader, num_classes, class_weights = get_isic_dataloaders\(batch_size=32\).*?model\.classifier\[0\]\.bias\[1\] = 0\.0',
    main_model_init,
    code, flags=re.DOTALL
)

trainer_train_epoch = r'''    def train_epoch(self, epoch, dataloader, device, print_interval=200):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 3. High-Resolution & Memory Management: Gradient Accumulation (steps=4)
        accumulation_steps = 4 
        self.optimizer.zero_grad()
        
        for batch_idx, data in enumerate(dataloader):
            if len(data) == 3:
                inputs, tab_inputs, targets = data
                inputs, tab_inputs, targets = inputs.to(device), tab_inputs.to(device), targets.to(device)
            else:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                tab_inputs = None
            
            with torch.cuda.amp.autocast():
                evidence = self.model(inputs, tab_inputs)
                loss = self.criterion(evidence, targets, epoch)
                loss = loss / accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * accumulation_steps'''

code = re.sub(
    r'    def train_epoch\(self, epoch, dataloader, device, print_interval=200\):.*?total_loss \+= loss\.item\(\)',
    trainer_train_epoch,
    code, flags=re.DOTALL
)

evaluate_forward = r'''    for data in test_loader:
        if len(data) == 3:
            inputs, tab_inputs, targets = data
            inputs, tab_inputs, targets = inputs.to(device), tab_inputs.to(device), targets.to(device)
        else:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            tab_inputs = None
            
        evidence = model(inputs, tab_inputs)'''

code = re.sub(
    r'    for inputs, targets in test_loader:.*?evidence = model\(inputs\)',
    evaluate_forward,
    code, flags=re.DOTALL
)

with open('c:/Users/ADMIN/Downloads/DesNet121/mdep_densenet_refactored.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Success')
