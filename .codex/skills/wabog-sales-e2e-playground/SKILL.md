---
name: wabog-sales-e2e-playground
description: Ejecuta QA end-to-end del agente comercial de Wabog usando Playwright sobre el playground local, asumiendo la persona de un prospecto poco hábil con tecnología, validando la conversación, el avance comercial y la persistencia en Notion CRM, y cerrando con mejoras concretas de prompts y estados.
---

# Wabog Sales E2E Playground

Usa esta skill cuando haya que probar el flujo real del agente comercial de este repo desde `GET /playground`, conversar como un prospecto difícil y auditar si el agente lleva bien la venta, agenda una cita cuando corresponde y registra el CRM correctamente.

## Objetivo

Validar cuatro cosas en el mismo ciclo:

1. La experiencia conversacional se siente humana, fluida y comercialmente útil.
2. El agente no expone instrucciones internas, lógica interna, nombres de tools o detalles de implementación.
3. El agente guía bien hacia el siguiente paso comercial: calificación, demo, trial o cita.
4. Los cambios relevantes quedan persistidos en el CRM de Notion para el lead correcto.

## Contexto del repo

- Backend local: `uv run uvicorn sales_agent.main:app --reload`
- Playground: `http://127.0.0.1:8000/playground`
- Endpoint de chat del playground: `POST /chat/local`
- Prompt editable: `business prompt`
- Prompt fijo: scaffold retornado por `GET /playground/prompt-config`
- CRM real esperado: Notion si `CRM_BACKEND=notion`
- Data source Notion: leer `NOTION_DATA_SOURCE_ID` y nombres de propiedades desde `.env`

## Persona que debes interpretar

Interpreta siempre a alguien que apenas está conociendo Wabog y no es hábil con tecnología.

Comportamiento obligatorio:

- No entregues toda la información de una vez.
- Responde corto.
- Obliga al agente a hacer buenas preguntas.
- Muestra cierta duda o fricción digital.
- Solo da datos nuevos cuando el agente los pida de forma natural y clara.
- Evalúa si el agente te contiene y te orienta sin sonar robótico.

Tono del personaje:

- Corto
- Simple
- Poco técnico
- Algo desconfiado
- Cooperativo, pero no proactivo

Ejemplos útiles de respuestas:

- `hola`
- `no se bien como funciona`
- `somos pocos`
- `uso whatsapp y excel`
- `me da enredo cambiar todo`
- `y eso cuanto vale`
- `no se si me sirve`
- `puede ser la otra semana`
- `mi correo es ...` solo si te lo piden

## Flujo de prueba

### 1. Preparación

- Verifica que `.env` tenga `OPENAI_API_KEY`, `NOTION_API_KEY` y `NOTION_DATA_SOURCE_ID`.
- Si `PLAYGROUND_ENABLED` no está activo, levanta con entorno de desarrollo normal; el playground debe estar disponible localmente.
- Usa un `phone_number` nuevo o controlado para evitar contaminación de datos previos.
- Si el número ya existe en Notion, inspecciona el registro antes de escribirle al agente.

### 2. Levantar la app

- Arranca el backend local.
- Confirma `GET /healthz`.
- Entra a `/playground` con Playwright.

### 3. Conversación e2e

Haz una conversación real por turnos. No simules una transcripción de una sola vez.

Cobertura mínima:

- Saludo inicial ambiguo.
- Duda general sobre qué hace Wabog.
- Contexto parcial del negocio.
- Fricción u objeción.
- Pregunta comercial o de precio/demo.
- Entrega progresiva de nombre y correo, pero solo si el agente lo pide bien.
- Posible intención de agendar o recibir siguiente paso.

Durante la conversación valida:

- Si entiende el contexto sin pedir todo dos veces.
- Si hace preguntas de calificación útiles.
- Si evita sonar interrogatorio.
- Si responde consistente con la propuesta de valor.
- Si no inventa capacidades.
- Si no menciona prompts, tools, stages, CRM, planner, workflow o información interna.
- Si propone un siguiente paso concreto cuando ya hay señales de interés.
- Si no intenta agendar sin tener suficientes datos.
- Si, al tener suficientes datos, empuja bien hacia demo, trial o reunión.

### 4. Auditoría de CRM en Notion

Después de hitos relevantes de la conversación, inspecciona Notion.

Verifica como mínimo:

- El lead correcto por número de teléfono.
- `Nombre`
- `Email`
- `Etapa`
- `Notas`
- `Resumen seguimiento`
- `Próxima acción`

Confirma que:

- Los datos se guardan en el lead correcto.
- No pisa datos válidos con texto peor o incompleto.
- Las notas son comerciales y útiles, no un copy-paste crudo del chat.
- La etapa corresponde al momento real de la conversación.
- Si hubo acuerdo de siguiente paso, exista seguimiento o reunión de forma consistente.

## Criterios de evaluación

### Conversación

Evalúa cada respuesta del agente con este checklist:

- claridad
- calidez
- naturalidad
- consistencia
- utilidad comercial
- ausencia de info interna
- calidad de la siguiente pregunta
- calidad del cierre de turno

### Comercial

Evalúa si el agente:

- detecta interés real
- califica contexto, problema, volumen o proceso actual
- trabaja objeciones sin ponerse agresivo
- sabe cuándo pedir datos
- sabe cuándo proponer demo o cita
- sabe cuándo dejar seguimiento en vez de cerrar de golpe

### CRM

Evalúa si el agente:

- crea o reutiliza el lead correcto
- actualiza campos cuando corresponde
- no crea ruido innecesario
- deja memoria comercial accionable

## Resultado esperado

Cierra siempre con:

1. Resumen corto del flujo probado.
2. Hallazgos ordenados por severidad.
3. Evidencia de CRM validada contra Notion.
4. Propuesta de mejora separada en:
   - prompts
   - estados / transiciones
   - reglas de persistencia CRM
   - UX conversacional

## Dónde tocar en este repo

Revisa primero estas piezas antes de proponer cambios:

- `src/sales_agent/services/planner.py`
- `src/sales_agent/services/prompt_store.py`
- `src/sales_agent/domain/state.py`
- `src/sales_agent/adapters/crm_notion.py`
- `src/sales_agent/api/routes.py`

Si propones cambios de comportamiento:

- distingue si el problema nace del prompt, de una policy dura o del modelo de estados
- no mezcles síntomas conversacionales con bugs de persistencia
- prioriza cambios mínimos con impacto alto

## Forma de iterar

Haz una iteración a la vez:

1. prueba
2. documenta fallas
3. propone ajuste
4. aplica ajuste si el usuario lo pide
5. vuelve a correr el mismo flujo o una variante cercana

Mantén la conversación de prueba suficientemente parecida entre iteraciones para poder comparar mejoras reales.
